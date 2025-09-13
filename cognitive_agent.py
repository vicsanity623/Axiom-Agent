# cognitive_agent.py

import uuid
import random
import json
import os
import re
from datetime import datetime
from functools import lru_cache
from graph_core import ConceptNode, RelationshipEdge, ConceptGraph
from knowledge_base import seed_domain_knowledge
from universal_interpreter import UniversalInterpreter
from dictionary_utils import get_word_info_from_wordnet

class CognitiveAgent:
    def __init__(self, brain_file="my_agent_brain.json", state_file="my_agent_state.json", 
                 load_from_file=True, brain_data=None, cache_data=None, inference_mode=False):
        
        print(f"Initializing Cognitive Agent...")
        self.brain_file = brain_file
        self.state_file = state_file
        self.inference_mode = inference_mode
        
        if self.inference_mode:
            print("   - Running in INFERENCE-ONLY mode. Learning is disabled.")

        self.interpreter = UniversalInterpreter()
        
        if load_from_file:
            print(f"   - Loading brain from file: {self.brain_file}")
            self.graph = ConceptGraph.load_from_file(self.brain_file)
            self._load_agent_state()
            
            # --- THE BOMBPROOF HEALTH CHECK ---
            # The most robust check is not if the 'axiom' node exists, but if the
            # critical relationship defining its name exists.
            agent_node = self.graph.get_node_by_name("agent")
            name_edge_exists = False
            if agent_node:
                name_edge = next((edge for edge in self.graph.get_edges_from_node(agent_node.id) if edge.type == "has_name"), None)
                if name_edge:
                    name_edge_exists = True
            
            if not name_edge_exists:
                print("   - CRITICAL FAILURE: Agent's core identity is missing. Re-seeding brain for integrity.")
                # We need a completely fresh graph object before seeding
                self.graph = ConceptGraph()
                seed_domain_knowledge(self)
                self.save_brain()
                self.save_state()
            # --- END OF BOMBPROOF HEALTH CHECK ---

        elif brain_data is not None and cache_data is not None:
            print("   - Initializing brain from loaded .axm model data.")
            self.graph = ConceptGraph.load_from_dict(brain_data)
            self.interpreter.interpretation_cache = dict(cache_data.get("interpretations", []))
            self.interpreter.synthesis_cache = dict(cache_data.get("synthesis", []))
            self.learning_iterations = brain_data.get("learning_iterations", 0)
        else:
            raise ValueError("Agent must be initialized with either files or data.")
            
        self.is_awaiting_clarification = False
        self.clarification_context = {}
        self.conversation_history = []
        self.enable_contextual_memory = False

    def _load_agent_state(self):
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    state_data = json.load(f)
                    self.learning_iterations = state_data.get("learning_iterations", 0)
                print(f"Agent state loaded from {self.state_file} (Learning Iterations: {self.learning_iterations}).")
            except Exception as e:
                self.learning_iterations = 0
        else:
            self.learning_iterations = 0

    def _save_agent_state(self):
        state_data = {"learning_iterations": self.learning_iterations}
        with open(self.state_file, 'w') as f:
            json.dump(state_data, f, indent=4)

    def _preprocess_self_reference(self, text: str) -> str:
        processed_text = re.sub(r'\byour name\b', "the agent's name", text, flags=re.IGNORECASE)
        processed_text = re.sub(r'\bwho are you\b', "what is the agent", processed_text, flags=re.IGNORECASE)
        processed_text = re.sub(r'\byou are\b', "the agent is", processed_text, flags=re.IGNORECASE)
        processed_text = re.sub(r'\byour\b', "the agent's", processed_text, flags=re.IGNORECASE)
        processed_text = re.sub(r'(?<!thank )\byou\b', "the agent", processed_text, flags=re.IGNORECASE)
        
        if processed_text != text:
            print(f"  [Pre-processor]: Normalized input to '{processed_text}'")
        return processed_text

    def chat(self, user_input: str) -> str:
        print(f"\nUser: {user_input}")
        self.graph.decay_activations()

        if self.is_awaiting_clarification:
            print("  [Curiosity]: Processing user's clarification...")
            interpretation = self.interpreter.interpret(user_input)
            entities = interpretation.get('entities', [])
            
            if entities:
                correct_answer_name = self._clean_phrase(entities[0]['name'])
                subject_name = self.clarification_context.get("subject")
                relation_type = self.clarification_context.get("conflicting_relation")
                subject_node = self.graph.get_node_by_name(subject_name)
                
                if subject_node and relation_type:
                    self._gather_facts_multihop.cache_clear()
                    print("  [Cache]: Cleared reasoning cache due to knowledge correction.")
                    for u, v, key, data in list(self.graph.graph.out_edges(subject_node.id, keys=True, data=True)):
                        if data.get('type') == relation_type:
                            target_node_data = self.graph.graph.nodes.get(v)
                            if target_node_data and target_node_data.get('name') == correct_answer_name:
                                self.graph.graph[u][v][key]['weight'] = 1.0
                                print(f"    - REINFORCED: {subject_name} --[{relation_type}]--> {correct_answer_name}")
                            else:
                                self.graph.graph[u][v][key]['weight'] = 0.1
                                if target_node_data:
                                    print(f"    - PUNISHED: {subject_name} --[{relation_type}]--> {target_node_data.get('name')}")
                    self.save_brain()
                
            self.is_awaiting_clarification = False
            self.clarification_context = {}
            final_response = "Thank you for the clarification. I have updated my knowledge."
            self.conversation_history.append(f"User: {user_input}")
            self.conversation_history.append(f"Agent: {final_response}")
            return final_response

        if self.enable_contextual_memory:
            interpretation = self.interpreter.interpret_with_context(user_input, self.conversation_history)
        else:
            normalized_input = self._preprocess_self_reference(user_input)
            interpretation = self.interpreter.interpret(normalized_input)
        
        print(f"  [Interpreter Output]: Intent='{interpretation.get('intent', 'N/A')}', "
              f"Entities={[e.get('name') for e in interpretation.get('entities', [])]}, "
              f"Relation={interpretation.get('relation')}")
        
        intent = interpretation.get('intent', 'unknown')
        entities = interpretation.get('entities', [])
        relation = interpretation.get('relation')
        
        structured_response = ""
        final_response = ""

        if intent == 'greeting': structured_response = "Hello User."
        elif intent == 'farewell': structured_response = "Goodbye User."
        elif intent in ['gratitude', 'acknowledgment']: structured_response = "You're welcome!"
        elif intent == 'positive_affirmation': structured_response = "I'm glad you think so!"
        elif intent == 'statement_of_fact' and relation:
            was_learned, response_message = self._process_statement_for_learning(relation)
            if not was_learned and self.is_awaiting_clarification:
                self.conversation_history.append(f"User: {user_input}")
                self.conversation_history.append(f"Agent: {response_message}")
                return response_message
            structured_response = response_message
        
        elif intent == 'statement_of_correction' and relation:
            print("  [Correction]: Processing user's correction...")
            subject_name = relation.get('subject')
            if isinstance(subject_name, dict): subject_name = subject_name.get('name')
            
            if subject_name:
                subject_node = self.graph.get_node_by_name(self._clean_phrase(subject_name))
                if subject_node:
                    self._gather_facts_multihop.cache_clear()
                    print("  [Cache]: Cleared reasoning cache due to knowledge correction.")
                    verb = relation.get('verb', '').lower().strip()
                    object_name = relation.get('object', '')
                    relation_type = self._get_relation_type(verb, subject_name, object_name)
                    for u, v, key, data in list(self.graph.graph.out_edges(subject_node.id, keys=True, data=True)):
                        if data.get('type') == relation_type:
                            old_fact_target_data = self.graph.graph.nodes.get(v)
                            if old_fact_target_data:
                                print(f"    - PUNISHING old fact: {subject_node.name} --[{data.get('type')}]--> {old_fact_target_data.get('name')}")
                                self.graph.graph[u][v][key]['weight'] = 0.1

            was_learned, response_message = self._process_statement_for_learning(relation)
            if was_learned:
                structured_response = "Thank you. I have corrected my knowledge."
            else:
                structured_response = f"I understood the correction, but failed to learn the new fact. Reason: {response_message}"

        elif intent == 'command' and 'show all facts' in user_input.lower():
            all_facts = []
            all_edges = [data for _, _, data in self.graph.graph.edges(data=True)]
            if not all_edges:
                structured_response = "My knowledge base is currently empty."
            else:
                sorted_edges = sorted(all_edges, key=lambda d: d.get('access_count', 0), reverse=True)
                for edge in sorted_edges:
                    if edge.get('type') == "might_relate": continue
                    source_node_data = self.graph.graph.nodes.get(edge.get('source'))
                    target_node_data = self.graph.graph.nodes.get(edge.get('target'))
                    if source_node_data and target_node_data:
                        fact_string = f"- {source_node_data.get('name').capitalize()} --[{edge.get('type')}]--> {target_node_data.get('name').capitalize()} (Salience: {edge.get('access_count', 0)})"
                        if edge.get('properties'):
                            fact_string += f" (Properties: {json.dumps(edge.get('properties'))})"
                        all_facts.append(fact_string)
                if all_facts:
                    structured_response = "Here are all the facts I have learned (most salient first):\n\n" + "\n".join(all_facts)
                else:
                    structured_response = "My knowledge base has concepts but no learned high-confidence facts."
        
        elif intent in ['question_about_entity', 'question_about_concept']:
            entity_name = (entities[0]['name'] if entities else user_input)
            clean_entity_name = self._clean_phrase(entity_name)

            if "agent" in clean_entity_name and "name" in user_input.lower():
                agent_node = self.graph.get_node_by_name("agent")
                if agent_node:
                    name_edge = next((edge for edge in self.graph.get_edges_from_node(agent_node.id) if edge.type == "has_name"), None)
                    if name_edge:
                        name_node_data = self.graph.graph.nodes.get(name_edge.target)
                        if name_node_data:
                            structured_response = f"My name is {name_node_data.get('name').capitalize()}."
                        else:
                            structured_response = "I know I have a name, but I can't seem to recall it right now."
                    else:
                        structured_response = "I don't have a name yet."
                else:
                    structured_response = "I don't seem to have a concept of myself right now."
            else:
                subject_node = self.graph.get_node_by_name(clean_entity_name)
                if not subject_node:
                    structured_response = f"I don't have any information about {entity_name}."
                else:
                    print(f"  [CognitiveAgent]: Starting reasoning for '{entity_name}'.")
                    facts_with_props = self._gather_facts_multihop(subject_node.id, max_hops=4)
                    
                    is_temporal_query = any(keyword in user_input.lower() for keyword in ['now', 'currently', 'today', 'this year'])
                    if is_temporal_query:
                        facts = self._filter_facts_for_temporal_query(facts_with_props)
                    else:
                        facts = {fact_str for fact_str, props_tuple in facts_with_props}

                    if not facts:
                        structured_response = f"I know the concept of {subject_node.name.capitalize()}, but I don't have any specific details for that query."
                    else:
                        structured_response = ". ".join(sorted(list(facts))) + "."
            
        else: structured_response = "I'm not sure how to process that. Could you rephrase?"
        
        non_synthesize_triggers = ["Hello User", "Goodbye User", "I understand. I have noted that.", "I don't have any information about", "My name is", "I know about", "That's an interesting topic about", "I'm not sure I fully understood that", "You're welcome!", "I'm glad you think so!", "Here are all the high-confidence facts I have learned", "Thank you for the clarification. I have updated my knowledge.", "Thank you. I have corrected my knowledge.", "I am currently in a read-only mode"]
        if any(trigger in structured_response for trigger in non_synthesize_triggers):
            final_response = structured_response
        else:
            print(f"  [Structured Response]: {structured_response}")
            fluent_response = self.interpreter.synthesize(structured_response, original_question=user_input)
            print(f"  [Synthesized Response]: {fluent_response}")
            final_response = fluent_response
            
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
        
        self.conversation_history.append(f"User: {user_input}")
        self.conversation_history.append(f"Agent: {final_response}")

        return final_response

    @lru_cache(maxsize=256)
    def _gather_facts_multihop(self, start_node_id: str, max_hops: int) -> tuple:
        print(f"  [Cache]: MISS! Executing full multi-hop graph traversal for node ID: {start_node_id}")
        
        start_node_data = self.graph.graph.nodes.get(start_node_id)
        if not start_node_data: return tuple()

        found_facts = {}
        queue = [(start_node_id, 0)]
        visited = {start_node_id}

        while queue:
            current_node_id, current_hop = queue.pop(0)
            if current_hop >= max_hops: continue
            
            current_node_data = self.graph.graph.nodes.get(current_node_id)
            if not current_node_data: continue

            for edge in self.graph.get_edges_from_node(current_node_id):
                if edge.type == "might_relate": continue
                target_node_data = self.graph.graph.nodes.get(edge.target)
                if target_node_data:
                    fact_str = f"{current_node_data.get('name').capitalize()} {edge.type.replace('_', ' ')} {target_node_data.get('name').capitalize()}"
                    if fact_str not in found_facts:
                        found_facts[fact_str] = edge
                    if edge.target not in visited:
                        visited.add(edge.target)
                        queue.append((edge.target, current_hop + 1))

            for edge in self.graph.get_edges_to_node(current_node_id):
                if edge.type == "might_relate": continue
                source_node_data = self.graph.graph.nodes.get(edge.source)
                if source_node_data:
                    fact_str = f"{source_node_data.get('name').capitalize()} {edge.type.replace('_', ' ')} {current_node_data.get('name').capitalize()}"
                    if fact_str not in found_facts:
                        found_facts[fact_str] = edge
                    if edge.source not in visited:
                        visited.add(edge.source)
                        queue.append((edge.source, current_hop + 1))
        
        all_facts_items = list(found_facts.items())

        if len(all_facts_items) > 10:
            original_subject = start_node_data.get('name', '').capitalize()
            relevance_filtered = [(f, e) for f, e in all_facts_items if f.startswith(original_subject)]
            if relevance_filtered: all_facts_items = relevance_filtered
        
        if len(all_facts_items) > 10:
            all_facts_items.sort(key=lambda item: item[1].access_count, reverse=True)
            all_facts_items = all_facts_items[:10]

        final_fact_tuples = tuple((fact_str, tuple(sorted(edge.properties.items()))) for fact_str, edge in all_facts_items)
        return final_fact_tuples
    
    def _filter_facts_for_temporal_query(self, facts_with_props_tuple: tuple) -> set:
        print("  [TemporalReasoning]: Filtering facts by date...")
        today = datetime.utcnow().date()
        best_fact = None
        best_date = None
        
        facts_list = [(fact_str, dict(props_tuple)) for fact_str, props_tuple in facts_with_props_tuple]

        for fact_str, props in facts_list:
            date_str = props.get('effective_date')
            if date_str:
                try:
                    fact_date = datetime.fromisoformat(date_str).date()
                    if fact_date <= today:
                        if best_date is None or fact_date > best_date:
                            best_date = fact_date
                            best_fact = fact_str
                except (ValueError, TypeError):
                    continue
        if best_fact:
            return {best_fact}
        else:
            return {fact_str for fact_str, props in facts_list if not props.get('effective_date')}
        
    def _clean_phrase(self, phrase: str) -> str:
        words = phrase.lower().split()
        if words and words[0] in ['a', 'an', 'the']: words = words[1:]
        return " ".join(words).strip()

    def _process_statement_for_learning(self, relation: dict) -> tuple[bool, str]:
        if self.inference_mode:
            return (False, "I am currently in a read-only mode and cannot learn new facts.")
        
        subject = relation.get('subject')
        verb = relation.get('verb')
        object_ = relation.get('object')
        properties = relation.get('properties', {})
        if not all([subject, verb, object_]): return (False, "I couldn't understand the structure of that fact.")
        
        subject_name = subject if isinstance(subject, str) else subject.get('name')
        if not subject_name: return (False, "Could not determine the subject of the fact.")
        
        objects_to_process = []
        if isinstance(object_, list):
            for item in object_:
                if isinstance(item, dict):
                    name = item.get('entity') or item.get('name')
                    if name: objects_to_process.append(name)
                elif isinstance(item, str):
                    objects_to_process.append(item)
        elif isinstance(object_, dict):
            name = object_.get('name')
            if name: objects_to_process.append(name)
        elif isinstance(object_, str):
            objects_to_process.append(object_)
        
        if not objects_to_process:
            return (False, "Could not determine the object(s) of the fact.")

        print(f"  [AGENT LEARNING: Processing interpreted statement: {subject_name} -> {verb} -> {objects_to_process}]")
        self.learning_iterations += 1
        learned_at_least_one = False
        
        for object_name in objects_to_process:
            verb_cleaned = verb.lower().strip()
            sub_node = self._add_or_update_concept(subject_name)
            relation_type = self._get_relation_type(verb_cleaned, subject_name, object_name)
            
            exclusive_relations = ["has_name", "is_capital_of", "is_located_in"]

            if relation_type in exclusive_relations and sub_node:
                for edge in self.graph.get_edges_from_node(sub_node.id):
                    if edge.type == relation_type:
                        existing_target_data = self.graph.graph.nodes[edge.target]
                        if existing_target_data.get('name') != self._clean_phrase(object_name):
                            print(f"  [Curiosity]: CONTRADICTION DETECTED (Exclusive Relationship)!")
                            conflicting_facts_str = (f"Fact 1: {sub_node.name} {edge.type.replace('_',' ')} {existing_target_data.get('name')}. " f"Fact 2: {sub_node.name} {relation_type.replace('_',' ')} {object_name}.")
                            question = self.interpreter.synthesize(conflicting_facts_str, mode="clarification_question")
                            self.is_awaiting_clarification = True
                            self.clarification_context = {"subject": sub_node.name, "conflicting_relation": relation_type}
                            return (False, question)

            obj_node = self._add_or_update_concept(object_name)
            if sub_node and obj_node:
                fact_already_exists = False
                for edge in self.graph.get_edges_from_node(sub_node.id):
                    if edge.type == relation_type and edge.target == obj_node.id:
                        print(f"    - Fact already exists: {sub_node.name} --[{relation_type}]--> {obj_node.name}")
                        fact_already_exists = True
                        break
                
                if not fact_already_exists:
                    self.graph.add_edge(sub_node, obj_node, relation_type, 0.9, properties=properties)
                    print(f"    Learned new fact: {sub_node.name} --[{relation_type}]--> {obj_node.name} with properties {properties}")
                    learned_at_least_one = True
        
        if learned_at_least_one:
            self._gather_facts_multihop.cache_clear()
            print("  [Cache]: Cleared reasoning cache due to new knowledge.")
            self.save_brain(); self.save_state()
            return (True, "I understand. I have noted that.")
        
        return (True, "I have processed that information.")
    
    def _get_relation_type(self, verb: str, subject: str, object_: str) -> str:
        if 'agent' in subject.lower() and verb in ['be', 'is', 'are', 'is named', 'is_named']:
            if len(object_.split()) == 1 and object_[0].isupper():
                return "has_name"
        relation_type_map = {"be": "is_a", "is": "is_a", "are": "is_a", "cause": "causes", "causes": "causes", "locate_in": "is_located_in", "located_in": "is_located_in", "part_of": "is_part_of", "learn": "learns", "release": "released", "released": "released"}
        return relation_type_map.get(verb, verb.replace(' ', '_'))

    def learn_new_fact_autonomously(self, fact_sentence: str) -> bool:
        if self.inference_mode:
            print("[Autonomous Learning]: Skipped. Agent is in inference mode.")
            return False
        print(f"[Autonomous Learning]: Attempting to learn fact: '{fact_sentence}'")
        interpretation = self.interpreter.interpret(fact_sentence)
        relation = interpretation.get('relation')
        print(f"  [Autonomous Learning]: Interpreted Relation: {relation}")
        if interpretation.get('intent') == 'statement_of_fact' and relation:
            was_learned, response_message = self._process_statement_for_learning(relation)
            if was_learned:
                print("[Autonomous Learning]: Successfully learned and saved new fact.")
                return True
            else:
                print(f"[Autonomous Learning]: Failed to process fact. Reason: {response_message}")
        else:
            print("[Autonomous Learning]: Could not interpret the sentence as a statement of fact.")
        return False

    def _add_or_update_concept(self, name: str, node_type: str = "concept"):
        clean_name = self._clean_phrase(name)
        if not clean_name: return None
        node = self.graph.get_node_by_name(clean_name)
        if not node:
            if len(clean_name.split()) > 1:
                determined_type = 'proper_noun' if any(c.isupper() for c in name) else 'noun_phrase'
            else:
                determined_type = get_word_info_from_wordnet(clean_name).get('type', node_type)
            node = self.graph.add_node(ConceptNode(clean_name, node_type=determined_type))
            print(f"    Added new concept to graph: {clean_name} ({node.type})")
        return node
    
    def manual_add_knowledge(self, concept_name1: str, concept_type1: str, relation: str, concept_name2: str, weight: float = 0.5):
        node1 = self._add_or_update_concept(concept_name1, node_type=concept_type1)
        node2 = self._add_or_update_concept(concept_name2) 
        if node1 and node2:
            self.graph.add_edge(node1, node2, relation, weight)
            print(f"Manually added knowledge: {concept_name1} --[{relation}]--> {concept_name2}")

    def save_brain(self):
        if not self.inference_mode:
            self.graph.save_to_file(self.brain_file)
    
    def save_state(self):
        if not self.inference_mode:
            self._save_agent_state()