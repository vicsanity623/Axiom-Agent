# cognitive_agent.py

import uuid
import random
import json
import os
import re
from graph_core import ConceptNode, RelationshipEdge, ConceptGraph
from knowledge_base import seed_domain_knowledge
from universal_interpreter import UniversalInterpreter
from dictionary_utils import get_word_info_from_wordnet

class CognitiveAgent:
    def __init__(self, brain_file="my_agent_brain.json", state_file="my_agent_state.json"):
        print(f"Initializing Cognitive Agent...")
        self.brain_file = brain_file
        self.state_file = state_file
        self.graph = ConceptGraph.load_from_file(self.brain_file)
        self.interpreter = UniversalInterpreter()
        self._load_agent_state()
        self.is_awaiting_clarification = False
        self.clarification_context = {}
        if not self.graph.get_node_by_name("apple"): 
            seed_domain_knowledge(self)
            self.save_brain()
            self.save_state()

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
        processed_text = re.sub(r'(?<!thank )\byou\b', "the agent", processed_text, flags=re.IGNORECASE)
        if processed_text != text:
            print(f"  [Pre-processor]: Normalized input to '{processed_text}'")
        return processed_text

    def chat(self, user_input: str) -> str:
        print(f"\nUser: {user_input}")
        self.graph.decay_activations()

        # --- NEW: Phase 3 - Learning from Clarification ---
        if self.is_awaiting_clarification:
            print("  [Curiosity]: Processing user's clarification...")
            # Use the interpreter to find the key entity in the user's answer
            interpretation = self.interpreter.interpret(user_input)
            entities = interpretation.get('entities', [])
            
            if entities:
                # Assume the first entity in the clarification is the correct answer
                correct_answer_name = self._clean_phrase(entities[0]['name'])
                subject_name = self.clarification_context.get("subject")
                relation_type = self.clarification_context.get("conflicting_relation")
                
                subject_node = self.graph.get_node_by_name(subject_name)
                
                if subject_node and relation_type:
                    # Find all conflicting facts
                    for edge in self.graph.get_edges_from_node(subject_node.id):
                        if edge.type == relation_type:
                            target_node = self.graph.nodes[edge.target]
                            if target_node.name == correct_answer_name:
                                # This is the correct fact, reinforce it
                                edge.weight = 1.0
                                print(f"    - REINFORCED: {subject_name} --[{relation_type}]--> {correct_answer_name}")
                            else:
                                # This is an incorrect fact, punish it
                                edge.weight = 0.1
                                print(f"    - PUNISHED: {subject_name} --[{relation_type}]--> {target_node.name}")
                    
                    self.save_brain() # Save the updated knowledge
                
            # Reset the clarification state
            self.is_awaiting_clarification = False
            self.clarification_context = {}
            return "Thank you for the clarification. I have updated my knowledge."
        # --- END OF NEW BLOCK ---

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
        elif intent == 'gratitude': structured_response = "You're welcome!"
        elif intent == 'positive_affirmation': structured_response = "I'm glad you think so!"
        elif intent == 'statement_of_fact' and relation:
            was_learned, response_message = self._process_statement_for_learning(relation)
            structured_response = response_message
        
        elif intent == 'command' and 'show all facts' in user_input.lower():
            all_facts = []
            if not self.graph.edges:
                structured_response = "My knowledge base is currently empty."
            else:
                for edge in self.graph.edges.values():
                    if edge.type == "might_relate": continue
                    source_node = self.graph.nodes.get(edge.source)
                    target_node = self.graph.nodes.get(edge.target)
                    if source_node and target_node:
                        fact_string = f"- {source_node.name.capitalize()} --[{edge.type}]--> {target_node.name.capitalize()}"
                        if edge.properties:
                            fact_string += f" (Properties: {json.dumps(edge.properties)})"
                        all_facts.append(fact_string)
                if all_facts:
                    structured_response = "Here are all the high-confidence facts I have learned:\n\n" + "\n".join(sorted(all_facts))
                else:
                    structured_response = "My knowledge base has concepts but no learned high-confidence facts."
        
        elif intent == 'question_about_entity' and entities:
            entity_name = entities[0]['name']
            if "agent" in entity_name.lower():
                agent_node = self.graph.get_node_by_name("agent")
                if "name" in normalized_input:
                    name_edge = next((edge for edge in self.graph.get_edges_from_node(agent_node.id) if edge.type == "has_name"), None) if agent_node else None
                    if name_edge:
                        name_node = self.graph.nodes[name_edge.target]
                        structured_response = f"My name is {name_node.name.capitalize()}."
                    else:
                        structured_response = "I don't have a name yet."
                else:
                    facts_to_synthesize = []
                    all_relations = self.graph.get_edges_from_node(agent_node.id) if agent_node else []
                    for edge in all_relations:
                        if edge.type == "is_a": facts_to_synthesize.append(f"I am a {self.graph.nodes[edge.target].name}.")
                    for edge in all_relations:
                        if edge.type == "has_name": facts_to_synthesize.append(f"My name is {self.graph.nodes[edge.target].name.capitalize()}.")
                    for edge in all_relations:
                        if edge.type == "can_do": facts_to_synthesize.append(f"I can {self.graph.nodes[edge.target].name}.")
                    if facts_to_synthesize:
                        structured_response = " ".join(facts_to_synthesize)
                    else:
                        structured_response = "I am an AI assistant designed to learn."
            else:
                print(f"  [CognitiveAgent]: Starting multi-hop reasoning for '{entity_name}'...")
                clean_entity_name = self._clean_phrase(entity_name)
                subject_node = self.graph.get_node_by_name(clean_entity_name)
                if not subject_node:
                    structured_response = f"I don't have any information about {entity_name}."
                else:
                    facts = self._gather_facts_multihop(subject_node, max_hops=2)
                    if not facts:
                        structured_response = f"I know the concept of {subject_node.name.capitalize()}, but I don't have any specific details."
                    else:
                        structured_response = ". ".join(sorted(list(facts))) + "."
        else: structured_response = "I'm not sure how to process that. Could you rephrase?"
        
        non_synthesize_triggers = [
            "Hello User", "Goodbye User", "I understand. I have noted that.",
            "I don't have any information about", "My name is", "I know about",
            "That's an interesting topic about", "I'm not sure I fully understood that",
            "You're welcome!", "I'm glad you think so!",
            "Here are all the high-confidence facts I have learned",
            "I have conflicting information"
        ]
        if any(trigger in structured_response for trigger in non_synthesize_triggers):
            final_response = structured_response
        else:
            print(f"  [Structured Response]: {structured_response}")
            fluent_response = self.interpreter.synthesize(structured_response, original_question=user_input)
            print(f"  [Synthesized Response]: {fluent_response}")
            final_response = fluent_response
            
        return final_response

    def _gather_facts_multihop(self, start_node: ConceptNode, max_hops: int) -> set:
        facts = set()
        queue = [(start_node.id, 0)]
        visited = {start_node.id}
        while queue:
            current_node_id, current_hop = queue.pop(0)
            if current_hop >= max_hops: continue
            current_node = self.graph.nodes.get(current_node_id)
            if not current_node: continue
            for edge in self.graph.get_edges_from_node(current_node_id):
                if edge.type == "might_relate": continue
                target_node = self.graph.nodes.get(edge.target)
                if target_node:
                    facts.add(f"{current_node.name.capitalize()} {edge.type.replace('_', ' ')} {target_node.name.capitalize()}")
                    if edge.target not in visited:
                        visited.add(edge.target)
                        queue.append((edge.target, current_hop + 1))
            for edge in self.graph.get_edges_to_node(current_node_id):
                if edge.type == "might_relate": continue
                source_node = self.graph.nodes.get(edge.source)
                if source_node:
                    fact_str = f"{source_node.name.capitalize()} {edge.type.replace('_', ' ')} {current_node.name.capitalize()}"
                    if edge.type == "is_a" and edge.weight > 0.8:
                        fact_str = f"{source_node.name.capitalize()} is also known as {current_node.name.capitalize()}"
                    facts.add(fact_str)
                    if edge.source not in visited:
                        visited.add(edge.source)
                        queue.append((edge.source, current_hop + 1))
        return facts
        
    def _clean_phrase(self, phrase: str) -> str:
        words = phrase.lower().split()
        if words and words[0] in ['a', 'an', 'the']: words = words[1:]
        return " ".join(words).strip()

    def _process_statement_for_learning(self, relation: dict) -> tuple[bool, str]:
        subject = relation.get('subject')
        verb = relation.get('verb')
        object_ = relation.get('object')
        properties = relation.get('properties', {})
        
        if not all([subject, verb, object_]): return (False, "I couldn't understand the structure of that fact.")

        subject_name = subject if isinstance(subject, str) else subject.get('name')
        object_name = object_ if isinstance(object_, str) else object_.get('name')
        
        if not all([subject_name, verb, object_name]): return (False, "I couldn't understand the structure of that fact.")

        print(f"  [AGENT LEARNING: Processing interpreted statement: {subject_name} -> {verb} -> {object_name}]")
        self.learning_iterations += 1
        verb_cleaned = verb.lower().strip()

        sub_node = self._add_or_update_concept(subject_name)
        relation_type = self._get_relation_type(verb_cleaned, subject_name, object_name)
        
        definitional_verbs = ["is_a", "has_property", "is_located_in", "has_name"]
        if relation_type in definitional_verbs:
            for edge in self.graph.get_edges_from_node(sub_node.id):
                if edge.type == relation_type and self.graph.nodes[edge.target].name != self._clean_phrase(object_name):
                    existing_target_node = self.graph.nodes[edge.target]
                    print(f"  [Curiosity]: CONTRADICTION DETECTED!")
                    print(f"    - Existing Fact: {sub_node.name} --[{edge.type}]--> {existing_target_node.name}")
                    print(f"    - New Fact:      {sub_node.name} --[{relation_type}]--> {object_name}")
                    
                    conflicting_facts_str = (
                        f"Fact 1: {sub_node.name} {edge.type.replace('_',' ')} {existing_target_node.name}. "
                        f"Fact 2: {sub_node.name} {relation_type.replace('_',' ')} {object_name}."
                    )
                    question = self.interpreter.synthesize(conflicting_facts_str, mode="clarification_question")
                    
                    self.is_awaiting_clarification = True
                    self.clarification_context = {"subject": sub_node.name, "conflicting_relation": relation_type}
                    
                    return (False, question)
        
        obj_node = self._add_or_update_concept(object_name)
        if sub_node and obj_node:
            self.graph.add_edge(sub_node, obj_node, relation_type, 0.9, properties=properties)
            print(f"    Learned new fact: {sub_node.name} --[{relation_type}]--> {obj_node.name} with properties {properties}")
            self.save_brain(); self.save_state()
            return (True, "I understand. I have noted that.")
        
        return (False, "I couldn't establish a clear fact from that.")
    
    def _get_relation_type(self, verb: str, subject: str, object_: str) -> str:
        if 'agent' in subject.lower() and verb in ['be', 'is', 'are', 'is named', 'is_named']:
            if len(object_.split()) == 1 and object_[0].isupper():
                return "has_name"
        
        relation_type_map = {
            "be": "is_a", "is": "is_a", "are": "is_a",
            "cause": "causes", "causes": "causes",
            "locate_in": "is_located_in", "located_in": "is_located_in",
            "part_of": "is_part_of", "learn": "learns",
            "release": "released", "released": "released"
        }
        return relation_type_map.get(verb, verb.replace(' ', '_'))

    def learn_new_fact_autonomously(self, fact_sentence: str) -> bool:
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
            node = self.graph.add_node(ConceptNode(clean_name, node_type=node_type))
            print(f"    Added new concept to graph: {clean_name} ({node.type})")
        return node
    
    def manual_add_knowledge(self, concept_name1: str, concept_type1: str, relation: str, concept_name2: str, weight: float = 0.5):
        node1 = self._add_or_update_concept(concept_name1, node_type=concept_type1)
        node2 = self._add_or_update_concept(concept_name2) 
        if node1 and node2:
            self.graph.add_edge(node1, node2, relation, weight)
            print(f"Manually added knowledge: {concept_name1} --[{relation}]--> {concept_name2}")

    def save_brain(self):
        self.graph.save_to_file(self.brain_file)
    
    def save_state(self):
        self._save_agent_state()