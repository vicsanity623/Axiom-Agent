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
        
        normalized_input = self._preprocess_self_reference(user_input)
        interpretation = self.interpreter.interpret(normalized_input)
        
        print(f"  [Interpreter Output]: Intent='{interpretation.get('intent', 'N/A')}', "
              f"Entities={[e.get('name') for e in interpretation.get('entities', [])]}, "
              f"Relation={interpretation.get('relation')}")
        
        intent = interpretation.get('intent', 'unknown')
        entities = interpretation.get('entities', [])
        relation = interpretation.get('relation')
        key_topics = interpretation.get('key_topics', [])
        structured_response = ""

        if intent == 'greeting': structured_response = "Hello User."
        elif intent == 'farewell': structured_response = "Goodbye User."
        elif intent == 'gratitude': structured_response = "You're welcome!"
        elif intent == 'positive_affirmation': structured_response = "I'm glad you think so!"
        elif intent == 'statement_of_fact' and relation:
            fact_learned = self._process_statement_for_learning(relation)
            if fact_learned: structured_response = "I understand. I have noted that."
            else: structured_response = "I couldn't establish a clear fact from that."
        
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
                    # --- FIX: Gather raw facts and delegate to the synthesizer ---
                    facts_to_synthesize = []
                    all_relations = self.graph.get_edges_from_node(agent_node.id) if agent_node else []
                    
                    # Prioritize the most important facts first for a better response
                    for edge in all_relations:
                        if edge.type == "is_a":
                            target_node = self.graph.nodes[edge.target]
                            facts_to_synthesize.append(f"I am a {target_node.name}.")
                    for edge in all_relations:
                        if edge.type == "has_name":
                            target_node = self.graph.nodes[edge.target]
                            facts_to_synthesize.append(f"My name is {target_node.name.capitalize()}.")
                    for edge in all_relations:
                        if edge.type == "can_do":
                            target_node = self.graph.nodes[edge.target]
                            facts_to_synthesize.append(f"I can {target_node.name}.")
                            
                    if facts_to_synthesize:
                        structured_response = " ".join(facts_to_synthesize)
                    else:
                        structured_response = "I am an AI assistant designed to learn."
                    # --- END FIX ---
            else:
                clean_entity_name = self._clean_phrase(entity_name)
                subject_node = self.graph.get_node_by_name(clean_entity_name)
                if not subject_node:
                    structured_response = f"I don't have any information about {entity_name}."
                else:
                    response_parts = [subject_node.name.capitalize()]
                    all_relations = self.graph.get_edges_from_node(subject_node.id)
                    for edge in sorted(all_relations, key=lambda e: e.type):
                        if edge.type == "might_relate": continue
                        target_node = self.graph.nodes[edge.target]
                        relation_verb = "is a" if edge.type == "is_a" else edge.type.replace('_', ' ')
                        response_parts.append(f"{relation_verb} {target_node.name.capitalize()}")
                    aliases = []
                    incoming_relations = self.graph.get_edges_to_node(subject_node.id)
                    for edge in incoming_relations:
                        if edge.type == "is_a" and edge.weight > 0.8:
                            alias_node = self.graph.nodes[edge.source]
                            aliases.append(alias_node.name.capitalize())
                    if aliases: response_parts.append(f"and is also known as {', '.join(aliases)}")
                    if len(response_parts) == 1: structured_response = f"I know about {subject_node.name.capitalize()}, but I don't have any specific details."
                    else: structured_response = " ".join(response_parts) + "."
        else: structured_response = "I'm not sure how to process that. Could you rephrase?"
        
        non_synthesize_triggers = [
            "Hello User", "Goodbye User", "I understand. I have noted that.",
            "I don't have any information about", "My name is", "I know about",
            "That's an interesting topic about", "I'm not sure I fully understood that",
            "You're welcome!", "I'm glad you think so!",
            "Here are all the high-confidence facts I have learned"
        ]
        if any(trigger in structured_response for trigger in non_synthesize_triggers):
            final_response = structured_response
        else:
            print(f"  [Structured Response]: {structured_response}")
            fluent_response = self.interpreter.synthesize(structured_response)
            print(f"  [Synthesized Response]: {fluent_response}")
            final_response = fluent_response
        return final_response

    def _clean_phrase(self, phrase: str) -> str:
        words = phrase.lower().split()
        if words and words[0] in ['a', 'an', 'the']: words = words[1:]
        return " ".join(words).strip()

    def _process_statement_for_learning(self, relation: dict) -> bool:
        subject = relation.get('subject')
        verb = relation.get('verb')
        object_ = relation.get('object')
        properties = relation.get('properties', {})
        
        if not all([subject, verb, object_]): return False

        subject_name = subject if isinstance(subject, str) else subject.get('name')
        object_name = object_ if isinstance(object_, str) else object_.get('name')
        
        if not all([subject_name, verb, object_name]):
            print(f"    [Agent Warning] Could not extract name from subject/object dict: {relation}")
            return False

        print(f"  [AGENT LEARNING: Processing interpreted statement: {subject_name} -> {verb} -> {object_name}]")
        self.learning_iterations += 1
        verb_cleaned = verb.lower().strip()

        if 'agent' in subject_name.lower() and verb_cleaned in ['be', 'is', 'are', 'is named', 'is_named']:
            if len(object_name.split()) == 1 and object_name[0].isupper():
                relation_type = "has_name"; subject_name = "agent"
            else: relation_type = "is_a"
        else:
            relation_type_map = {"be": "is_a", "is": "is_a", "are": "is_a", "cause": "causes", "causes": "causes", "locate_in": "is_located_in", "located_in": "is_located_in", "part_of": "is_part_of", "learn": "learns", "release": "released", "released": "released"}
            relation_type = relation_type_map.get(verb_cleaned, verb_cleaned.replace(' ', '_'))

        sub_node = self._add_or_update_concept(subject_name)
        obj_node = self._add_or_update_concept(object_name)
        if sub_node and obj_node:
            self.graph.add_edge(sub_node, obj_node, relation_type, 0.9, properties=properties)
            print(f"    Learned new fact: {sub_node.name} --[{relation_type}]--> {obj_node.name} with properties {properties}")
            self.save_brain(); self.save_state()
            return True
        return False
        
    def learn_new_fact_autonomously(self, fact_sentence: str) -> bool:
        print(f"[Autonomous Learning]: Attempting to learn fact: '{fact_sentence}'")
        interpretation = self.interpreter.interpret(fact_sentence)
        relation = interpretation.get('relation')
        print(f"  [Autonomous Learning]: Interpreted Relation: {relation}")
        if interpretation.get('intent') == 'statement_of_fact' and relation:
            fact_learned = self._process_statement_for_learning(relation)
            if fact_learned:
                print("[Autonomous Learning]: Successfully learned and saved new fact.")
                return True
            else:
                print("[Autonomous Learning]: Failed to process the interpreted fact.")
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