from __future__ import annotations

import time
from functools import lru_cache
from typing import TYPE_CHECKING

# knowledge_base.py
from tqdm import tqdm

from axiom.dictionary_utils import get_word_info_from_wordnet

get_word_info_from_wordnet = lru_cache(maxsize=None)(get_word_info_from_wordnet)

if TYPE_CHECKING:
    from axiom.cognitive_agent import CognitiveAgent

    from .universal_interpreter import PropertyData

LEXICON_PROMOTION_THRESHOLD = 0.8
LEXICON_OBSERVATION_DECAY = 0.0

ALL_KNOWLEDGE = [
    # --- Self-Identity Knowledge ---
    ("agent", "concept", "has_name", "Axiom", 1.0),
    ("agent", "concept", "is_a", "cognitive agent", 1.0),
    ("cognitive agent", "concept", "is_a", "AI assistant", 0.9),
    ("agent", "concept", "has_ability", "learn", 0.9),
    ("agent", "concept", "has_ability", "reason", 0.8),
    ("agent", "concept", "has_ability", "understand", 0.8),
    ("agent", "concept", "has_purpose", "assist users", 0.9),
    ("agent", "concept", "communicates_via", "text", 1.0),
    ("agent", "concept", "exists_on", "a server", 1.0),
    ("agent", "concept", "has_ability", "problem solving", 0.8),
    ("agent", "concept", "was_created_by", "humans", 0.9),
    ("agent", "concept", "operates_using", "algorithms", 0.8),
    ("agent", "concept", "has_ability", "improve over time", 0.7),
    ("agent", "concept", "requires", "computation", 0.9),
    ("agent", "concept", "has_capability", "natural language processing", 0.8),
    ("agent", "concept", "has_ability", "follow instructions", 0.9),
    ("agent", "concept", "has_ability", "access knowledge", 0.9),
    ("agent", "concept", "has_limitation", "no physical body", 0.8),
    ("agent", "concept", "functions_through", "code", 0.9),
    ("agent", "concept", "has_property", "long-term memory", 0.8),
    ("agent", "concept", "has_ability", "process information", 0.9),
    ("agent", "concept", "has_ability", "respond to queries", 0.9),
    ("agent", "concept", "operates_on", "electricity", 0.8),
    ("agent", "concept", "has_goal", "to be helpful", 0.8),
    ("agent", "concept", "has_ability", "explain concepts", 0.7),
    ("agent", "concept", "has_ability", "understand context", 0.7),
    ("agent", "concept", "has_ability", "summarize information", 0.7),
    ("agent", "concept", "has_limitation", "no emotions", 0.8),
    ("agent", "concept", "has_ability", "translate languages", 0.7),
    ("agent", "concept", "uses", "pattern recognition", 0.8),
    ("agent", "concept", "has_ability", "generate creative content", 0.7),
    ("agent", "concept", "has_property", "consistency in responses", 0.7),
    ("agent", "concept", "has_ability", "analyze data", 0.8),
    ("agent", "concept", "has_ability", "operate 24/7", 0.9),
    ("agent", "concept", "has_access_to", "vast information", 0.8),
    ("agent", "concept", "has_ability", "adapt to new topics", 0.7),
    ("agent", "concept", "has_ability", "follow ethical guidelines", 0.8),
    ("agent", "concept", "has_ability", "collaborate with humans", 0.7),
    ("agent", "concept", "has_property", "potential for continuous improvement", 0.7),
    # --- World Knowledge (Physics, Astronomy, etc.) ---
    ("sky", "concept", "has_property", "blue", 0.9),
    ("sun", "concept", "is_a", "star", 1.0),
    ("sun", "concept", "emits", "light", 1.0),
    ("water", "concept", "is_a", "liquid", 0.8),
    ("fire", "concept", "has_property", "hot", 1.0),
    ("moon", "concept", "reflects", "light", 0.9),
    ("gravity", "concept", "affects", "all objects", 0.9),
    ("rain", "concept", "is_composed_of", "water droplets", 0.9),
    ("wind", "concept", "is_a", "moving air", 0.9),
    ("ice", "concept", "is_a", "frozen water", 0.9),
    ("cloud", "concept", "is_made_of", "water vapor", 0.9),
    ("earth", "planet", "has_property", "gravity", 0.9),
    ("sound", "concept", "travels_through", "air", 0.8),
    ("light", "concept", "is_faster_than", "sound", 0.9),
    ("metal", "concept", "conducts", "electricity", 0.9),
    ("magnet", "concept", "attracts", "iron", 0.9),
    ("volcano", "concept", "erupts", "lava", 0.9),
    ("earthquake", "concept", "causes", "ground shaking", 0.9),
    ("tornado", "concept", "is_a", "violent storm", 0.9),
    ("atom", "concept", "is_a", "smallest unit of matter", 0.9),
    ("energy", "concept", "has_property", "cannot be created or destroyed", 0.8),
    ("friction", "concept", "causes", "heat", 0.8),
    ("vacuum", "concept", "has_property", "no air", 0.9),
    ("lightning", "concept", "is_a", "electrical discharge", 0.9),
    ("echo", "concept", "is_a", "reflected sound", 0.9),
    ("lens", "concept", "refracts", "light", 0.8),
    ("prism", "concept", "splits", "white light", 0.8),
    ("evaporation", "concept", "changes", "liquid to gas", 0.9),
    ("condensation", "concept", "changes", "gas to liquid", 0.9),
    ("orbit", "concept", "is_a", "path around a planet", 0.9),
    ("comet", "concept", "has_property", "a tail", 0.9),
    ("galaxy", "concept", "contains", "stars", 0.9),
    ("black hole", "concept", "has_property", "strong gravity", 0.8),
    ("supernova", "concept", "is_a", "exploding star", 0.8),
    ("nebula", "concept", "is_a", "cloud of gas and dust", 0.8),
    # --- Geography Knowledge ---
    ("earth", "planet", "has_satellite", "the moon", 1.0),
    ("paris", "city", "is_located_in", "france", 1.0),
    ("france", "country", "is_located_in", "europe", 1.0),
    ("mount everest", "mountain", "is_a", "tallest mountain", 1.0),
    ("london", "city", "is_capital_of", "england", 1.0),
    ("nile", "river", "is_located_in", "africa", 1.0),
    ("sahara", "desert", "is_located_in", "africa", 1.0),
    ("pacific", "ocean", "is_a", "largest ocean", 1.0),
    ("asia", "continent", "has_property", "highest population", 0.9),
    ("amazon", "river", "is_located_in", "south america", 1.0),
    ("new york", "city", "is_located_in", "united states", 1.0),
    ("tokyo", "city", "is_capital_of", "japan", 1.0),
    ("antarctica", "continent", "is_a", "coldest continent", 1.0),
    ("australia", "country", "is_a", "continent", 1.0),
    ("rome", "city", "is_capital_of", "italy", 1.0),
    ("ganges", "river", "is_located_in", "india", 1.0),
    ("himalayas", "mountain range", "is_located_in", "asia", 1.0),
    ("dead sea", "lake", "is_a", "lowest point on land", 1.0),
    ("greenland", "island", "is_a", "largest island", 1.0),
    ("madagascar", "island", "is_located_off", "africa", 1.0),
    ("siberia", "region", "is_located_in", "russia", 1.0),
    ("amazon rainforest", "forest", "is_located_in", "south america", 1.0),
    ("grand canyon", "canyon", "is_located_in", "arizona", 1.0),
    ("great barrier reef", "reef", "is_located_off", "australia", 1.0),
    ("yellowstone", "national park", "has_property", "geysers", 1.0),
    ("niagara falls", "waterfall", "is_located_on_border_of", "usa and canada", 1.0),
    ("mississippi", "river", "is_a", "longest river in usa", 1.0),
    ("alps", "mountain range", "is_located_in", "europe", 1.0),
    ("red sea", "sea", "is_located_between", "africa and asia", 1.0),
    ("caspian sea", "body of water", "is_a", "largest lake", 1.0),
    ("andes", "mountain range", "is_located_in", "south america", 1.0),
    ("kilimanjaro", "mountain", "is_located_in", "tanzania", 1.0),
    ("gobi", "desert", "is_located_in", "asia", 1.0),
    # --- Biology Knowledge ---
    ("human", "species", "is_a", "mammal", 1.0),
    ("dog", "species", "is_a", "mammal", 1.0),
    ("cat", "species", "is_a", "mammal", 1.0),
    ("mammal", "class", "is_a", "animal", 1.0),
    ("mammal", "class", "gives_birth_to", "live young", 0.95),
    ("animal", "kingdom", "is_a", "living thing", 1.0),
    ("tree", "plant", "is_a", "living thing", 1.0),
    ("fish", "animal", "lives_in", "water", 0.9),
    ("bird", "animal", "has_property", "feathers", 0.9),
    ("insect", "animal", "has_property", "six legs", 0.9),
    ("reptile", "animal", "is_a", "cold-blooded", 0.9),
    ("plant", "living thing", "produces", "oxygen", 0.9),
    ("mammal", "animal", "has_ability", "feed milk to young", 0.9),
    ("human", "mammal", "has_property", "opposable thumbs", 0.9),
    ("bat", "mammal", "has_ability", "fly", 0.9),
    ("amphibian", "animal", "lives_in", "land and water", 0.9),
    ("fungus", "living thing", "is_not_a", "plant or animal", 0.8),
    ("whale", "mammal", "is_a", "largest animal", 0.9),
    ("bee", "insect", "produces", "honey", 0.9),
    ("spider", "arachnid", "has_property", "eight legs", 0.9),
    ("butterfly", "insect", "undergoes", "metamorphosis", 0.9),
    ("cactus", "plant", "stores", "water", 0.9),
    ("venus flytrap", "plant", "eats", "insects", 0.9),
    ("coral", "animal", "builds", "reefs", 0.9),
    ("octopus", "animal", "has_property", "eight arms", 0.9),
    ("kangaroo", "marsupial", "has_property", "pouch", 0.9),
    ("penguin", "bird", "cannot", "fly", 0.9),
    ("elephant", "mammal", "has_property", "trunk", 0.9),
    ("giraffe", "mammal", "has_property", "long neck", 0.9),
    ("shark", "fish", "has_property", "cartilage skeleton", 0.9),
    ("snake", "reptile", "has_property", "no legs", 0.9),
    ("turtle", "reptile", "has_property", "shell", 0.9),
    ("frog", "amphibian", "undergoes", "metamorphosis", 0.9),
    ("algae", "organism", "lives_in", "water", 0.9),
    ("mushroom", "fungus", "reproduces_with", "spores", 0.9),
    ("bacteria", "microorganism", "is_a", "single-celled organism", 0.9),
    ("virus", "microorganism", "requires", "host to reproduce", 0.9),
    # --- Food Knowledge ---
    ("apple", "fruit", "is_a", "food", 0.9),
    ("banana", "fruit", "is_a", "food", 0.9),
    ("carrot", "vegetable", "is_a", "food", 0.9),
    ("bread", "food", "is_made_from", "flour", 0.9),
    ("cheese", "food", "is_made_from", "milk", 0.9),
    ("rice", "food", "is_a", "grain", 0.9),
    ("chicken", "food", "is_a", "meat", 0.9),
    ("chocolate", "food", "is_made_from", "cocoa", 0.9),
    ("pasta", "food", "is_made_from", "wheat", 0.9),
    ("soup", "food", "is_a", "liquid", 0.8),
    ("salad", "food", "contains", "vegetables", 0.9),
    ("ice cream", "food", "has_property", "cold", 0.9),
    ("pizza", "food", "has_component", "crust", 0.9),
    ("orange", "fruit", "is_a", "citrus fruit", 0.9),
    ("potato", "vegetable", "is_a", "root vegetable", 0.9),
    ("tomato", "fruit", "is_used_as", "a vegetable", 0.9),
    ("onion", "vegetable", "has_property", "layers", 0.9),
    ("garlic", "vegetable", "is_used_for", "flavoring", 0.9),
    ("beef", "meat", "comes_from", "cows", 0.9),
    ("pork", "meat", "comes_from", "pigs", 0.9),
    ("fish", "food", "is_a", "source of protein", 0.9),
    ("egg", "food", "comes_from", "chickens", 0.9),
    ("milk", "beverage", "comes_from", "cows", 0.9),
    ("yogurt", "food", "is_made_from", "milk", 0.9),
    ("honey", "food", "is_made_by", "bees", 0.9),
    ("coffee", "beverage", "is_made_from", "beans", 0.9),
    ("tea", "beverage", "is_made_from", "leaves", 0.9),
    ("sugar", "ingredient", "has_property", "sweet", 0.9),
    ("salt", "ingredient", "has_property", "salty", 0.9),
    ("flour", "ingredient", "is_made_from", "grains", 0.9),
    ("butter", "dairy", "is_made_from", "milk", 0.9),
    ("oil", "ingredient", "is_used_for", "cooking", 0.9),
    ("vinegar", "ingredient", "has_property", "sour", 0.9),
    # --- Abstract Concepts (Colors) ---
    ("color", "attribute", "is_a", "concept", 1.0),
    ("red", "descriptor", "is_a", "color", 0.9),
    ("green", "descriptor", "is_a", "color", 0.9),
    ("yellow", "descriptor", "is_a", "color", 0.9),
    ("blue", "descriptor", "is_a", "color", 0.9),
    ("orange", "descriptor", "is_a", "color", 0.9),
    ("purple", "descriptor", "is_a", "color", 0.9),
    ("black", "descriptor", "is_a", "color", 0.9),
    ("white", "descriptor", "is_a", "color", 0.9),
    ("brown", "descriptor", "is_a", "color", 0.9),
    ("pink", "descriptor", "is_a", "color", 0.9),
    ("gray", "descriptor", "is_a", "color", 0.9),
    ("cyan", "descriptor", "is_a", "color", 0.9),
    ("magenta", "descriptor", "is_a", "color", 0.9),
    ("gold", "descriptor", "is_a", "color", 0.9),
    ("silver", "descriptor", "is_a", "color", 0.9),
    ("beige", "descriptor", "is_a", "color", 0.9),
    ("turquoise", "descriptor", "is_a", "color", 0.9),
    ("lavender", "descriptor", "is_a", "color", 0.9),
    ("maroon", "descriptor", "is_a", "color", 0.9),
    ("navy", "descriptor", "is_a", "color", 0.9),
    ("olive", "descriptor", "is_a", "color", 0.9),
    ("teal", "descriptor", "is_a", "color", 0.9),
    ("coral", "descriptor", "is_a", "color", 0.9),
    ("indigo", "descriptor", "is_a", "color", 0.9),
    ("violet", "descriptor", "is_a", "color", 0.9),
    ("crimson", "descriptor", "is_a", "color", 0.9),
    ("khaki", "descriptor", "is_a", "color", 0.9),
    ("plum", "descriptor", "is_a", "color", 0.9),
    ("salmon", "descriptor", "is_a", "color", 0.9),
    ("tan", "descriptor", "is_a", "color", 0.9),
    ("mint", "descriptor", "is_a", "color", 0.9),
    # --- Abstract Concepts (Sentiments) ---
    ("sentiment", "attribute", "is_a", "concept", 1.0),
    ("happy", "descriptor", "is_a", "sentiment", 0.8),
    ("sad", "descriptor", "is_a", "sentiment", 0.8),
    ("angry", "descriptor", "is_a", "sentiment", 0.8),
    ("excited", "descriptor", "is_a", "sentiment", 0.8),
    ("fearful", "descriptor", "is_a", "sentiment", 0.8),
    ("surprised", "descriptor", "is_a", "sentiment", 0.8),
    ("disgusted", "descriptor", "is_a", "sentiment", 0.8),
    ("calm", "descriptor", "is_a", "sentiment", 0.8),
    ("confused", "descriptor", "is_a", "sentiment", 0.8),
    ("proud", "descriptor", "is_a", "sentiment", 0.8),
    ("jealous", "descriptor", "is_a", "sentiment", 0.8),
    ("anxious", "descriptor", "is_a", "sentiment", 0.8),
    ("content", "descriptor", "is_a", "sentiment", 0.8),
    ("curious", "descriptor", "is_a", "sentiment", 0.8),
    ("depressed", "descriptor", "is_a", "sentiment", 0.8),
    ("embarrassed", "descriptor", "is_a", "sentiment", 0.8),
    ("enthusiastic", "descriptor", "is_a", "sentiment", 0.8),
    ("frustrated", "descriptor", "is_a", "sentiment", 0.8),
    ("grateful", "descriptor", "is_a", "sentiment", 0.8),
    ("guilty", "descriptor", "is_a", "sentiment", 0.8),
    ("hopeful", "descriptor", "is_a", "sentiment", 0.8),
    ("impatient", "descriptor", "is_a", "sentiment", 0.8),
    ("inspired", "descriptor", "is_a", "sentiment", 0.8),
    ("lonely", "descriptor", "is_a", "sentiment", 0.8),
    ("nostalgic", "descriptor", "is_a", "sentiment", 0.8),
    ("optimistic", "descriptor", "is_a", "sentiment", 0.8),
    ("pessimistic", "descriptor", "is_a", "sentiment", 0.8),
    ("relieved", "descriptor", "is_a", "sentiment", 0.8),
    ("romantic", "descriptor", "is_a", "sentiment", 0.8),
    ("satisfied", "descriptor", "is_a", "sentiment", 0.8),
    ("sympathetic", "descriptor", "is_a", "sentiment", 0.8),
]


def seed_domain_knowledge(agent_instance: CognitiveAgent) -> None:
    """Seed the agent's brain with a foundational set of facts using progress bars."""
    print("   - Seeding a vast initial world knowledge base...")

    concepts_to_promote = {}
    for subject, s_type, relation, obj, weight in ALL_KNOWLEDGE:
        concepts_to_promote[subject] = s_type
        if obj not in concepts_to_promote:
            concepts_to_promote[obj] = "concept"

    for concept, concept_type in tqdm(
        concepts_to_promote.items(),
        desc="     - Seeding & Promoting Concepts",
    ):
        agent_instance.lexicon.add_linguistic_knowledge_quietly(concept, concept_type)

    for subject, s_type, relation, obj, weight in tqdm(
        ALL_KNOWLEDGE,
        desc="     - Seeding Relationships     ",
    ):
        agent_instance.manual_add_knowledge_quietly(
            subject,
            s_type,
            relation,
            obj,
            weight,
        )

    print("     - Integrating WordNet definitions for seeded concepts...")

    seeded_words = {
        data["name"]
        for _, data in agent_instance.graph.graph.nodes(data=True)
        if " " not in data["name"]
    }

    for word in tqdm(list(seeded_words), desc="     - Integrating WordNet     "):
        word_info = get_word_info_from_wordnet(word)
        if word_info["hypernyms_raw"]:
            main_node = agent_instance.graph.get_node_by_name(word)
            for hypernym_word in word_info["hypernyms_raw"][:1]:
                hypernym_node = agent_instance._add_or_update_concept_quietly(
                    hypernym_word,
                )
                if main_node and hypernym_node and main_node.id != hypernym_node.id:
                    agent_instance.graph.add_edge(main_node, hypernym_node, "is_a", 0.7)

    print("   - Vast domain knowledge seeding complete.")


def seed_core_vocabulary(agent_instance: CognitiveAgent) -> None:
    """Seed the agent's Lexicon with a foundational English vocabulary.

    This function teaches the agent the basic building blocks of English
    grammar. It populates the knowledge graph with nodes for common parts
    of speech, articles, verbs, prepositions, and conjunctions.

    This foundational knowledge is essential for the `SymbolicParser` to
    function and for the "Unknown Word" reflex to correctly identify
    substantive new words to learn.

    Args:
        agent_instance: The instance of the CognitiveAgent to be seeded.
    """
    print("     - Seeding core vocabulary for Lexicon...")
    core_vocab = {
        "noun": "concept",
        "verb": "concept",
        "adjective": "concept",
        "adverb": "concept",
        "pronoun": "concept",
        "preposition": "concept",
        "conjunction": "concept",
        "determiner": "concept",
        "article": "concept",
        "a": "article",
        "an": "article",
        "the": "article",
        "this": "determiner",
        "that": "determiner",
        "these": "determiner",
        "those": "determiner",
        "is": "verb",
        "are": "verb",
        "was": "verb",
        "were": "verb",
        "be": "verb",
        "being": "verb",
        "been": "verb",
        "has": "verb",
        "have": "verb",
        "had": "verb",
        "having": "verb",
        "do": "verb",
        "does": "verb",
        "did": "verb",
        "say": "verb",
        "says": "verb",
        "said": "verb",
        "go": "verb",
        "goes": "verb",
        "went": "verb",
        "get": "verb",
        "gets": "verb",
        "got": "verb",
        "make": "verb",
        "makes": "verb",
        "made": "verb",
        "know": "verb",
        "knows": "verb",
        "knew": "verb",
        "think": "verb",
        "thinks": "verb",
        "thought": "verb",
        "see": "verb",
        "sees": "verb",
        "saw": "verb",
        "come": "verb",
        "comes": "verb",
        "came": "verb",
        "take": "verb",
        "takes": "verb",
        "took": "verb",
        "give": "verb",
        "gives": "verb",
        "gave": "verb",
        "birth": "noun",
        "lay": "verb",
        "include": "verb",
        "includes": "verb",
        "consist": "verb",
        "consists": "verb",
        "lays": "verb",
        "i": "pronoun",
        "you": "pronoun",
        "he": "pronoun",
        "she": "pronoun",
        "it": "pronoun",
        "we": "pronoun",
        "they": "pronoun",
        "me": "pronoun",
        "him": "pronoun",
        "her": "pronoun",
        "us": "pronoun",
        "them": "pronoun",
        "of": "preposition",
        "in": "preposition",
        "to": "preposition",
        "for": "preposition",
        "with": "preposition",
        "on": "preposition",
        "at": "preposition",
        "from": "preposition",
        "by": "preposition",
        "about": "preposition",
        "as": "preposition",
        "into": "preposition",
        "like": "preposition",
        "through": "preposition",
        "after": "preposition",
        "over": "preposition",
        "between": "preposition",
        "out": "preposition",
        "against": "preposition",
        "during": "preposition",
        "without": "preposition",
        "before": "preposition",
        "under": "preposition",
        "around": "preposition",
        "among": "preposition",
        "and": "conjunction",
        "but": "conjunction",
        "or": "conjunction",
        "so": "conjunction",
        "if": "conjunction",
        "while": "conjunction",
        "because": "conjunction",
    }

    for word, pos in tqdm(core_vocab.items(), desc="     - Seeding lexicon         "):
        agent_instance.lexicon.add_linguistic_knowledge_quietly(word, pos)

    print("     - Core vocabulary seeding complete.")


def promote_word(
    agent_instance: CognitiveAgent,
    word: str,
    pos: str,
    confidence: float,
):
    """Directly promotes a word to a given part of speech."""
    node = agent_instance.graph.get_node_by_name(word)
    if not node:
        return

    props = agent_instance.graph.graph.nodes[node.id].setdefault("properties", {})
    props["lexical_promoted_as"] = pos
    props["lexical_promoted_confidence"] = confidence
    props.pop("lexical_observations", None)

    for rel, interp, ts in list(getattr(agent_instance, "pending_relations", [])):
        sub = (rel.get("subject") or "").lower()
        obj = (rel.get("object") or "").lower()
        if word in sub or word in obj:
            status = validate_and_add_relation(agent_instance, rel, interp)
            if status in ("inserted", "replaced", "contradiction_stored"):
                try:
                    agent_instance.pending_relations.remove((rel, interp, ts))
                except ValueError:
                    pass


def record_lexical_observation(
    agent_instance,
    word: str,
    observed_pos: str,
    confidence: float = 0.5,
):
    """
    Record a single POS observation for a `word`. This accumulates votes
    in the node's properties under `lexical_observations`.
    """
    if not word:
        return
    word = word.lower().strip()
    node = agent_instance.graph.get_node_by_name(word)
    if not node:
        node = agent_instance._add_or_update_concept_quietly(word)

    props = agent_instance.graph.graph.nodes[node.id].setdefault("properties", {})
    obs = props.setdefault("lexical_observations", {"votes": {}, "total": 0.0})

    votes = obs["votes"]
    votes[observed_pos] = float(votes.get(observed_pos, 0.0)) + float(confidence)
    obs["total"] = float(obs.get("total", 0.0)) + float(confidence)

    try_promote_lexicon(agent_instance, word)


def try_promote_lexicon(
    agent_instance,
    word: str,
    threshold: float = LEXICON_PROMOTION_THRESHOLD,
):
    """
    Promote the token to a stable lexicon POS once the top POS crosses the threshold.

    Creates an `is_a` edge word -> POS with properties marking provenance.
    """

    if not word:
        return False
    word = word.lower().strip()
    node = agent_instance.graph.get_node_by_name(word)
    if not node:
        return False

    props = agent_instance.graph.graph.nodes[node.id].get("properties", {})
    obs = props.get("lexical_observations")
    if not obs or float(obs.get("total", 0.0)) <= 0.0:
        return False

    votes = obs["votes"]
    total = float(obs["total"])
    top_pos, top_score = None, 0.0
    for pos, score in votes.items():
        if float(score) > top_score:
            top_pos, top_score = pos, float(score)

    if not top_pos:
        return False

    frac = top_score / total if total > 0 else 0.0
    if frac >= threshold:
        promote_word(agent_instance, word, top_pos, float(frac))

        node = agent_instance.graph.get_node_by_name(word)
        pos_node = agent_instance._add_or_update_concept_quietly(top_pos)
        if node and pos_node:
            agent_instance.graph.add_edge(node, pos_node, "is_a", float(frac))

        return True
    return False


def add_pending_relation(agent_instance, relation: dict, interpretation: PropertyData):
    """Store a relation temporarily until dependent lexicon entries are promoted."""

    agent_instance.pending_relations.append((relation, interpretation, time.time()))


def validate_and_add_relation(
    agent_instance,
    relation: dict,
    interpretation: PropertyData,
):
    """
    Validate a parsed relation, deferring if it contains un-promoted words
    with low confidence, and otherwise inserting it into the graph.
    """
    subject_name = (relation.get("subject") or "").strip().lower()
    object_name = (relation.get("object") or "").strip().lower()
    relation_type = (
        relation.get("predicate") or relation.get("verb") or relation.get("relation")
    )
    if not subject_name or not object_name or not relation_type:
        return "error"

    subj_node = agent_instance._add_or_update_concept_quietly(subject_name)
    obj_node = agent_instance._add_or_update_concept_quietly(object_name)
    if not subj_node or not obj_node:
        return "error"

    provenance = interpretation.get("provenance", "user")
    confidence = float(interpretation.get("confidence", 0.0))

    def is_node_promoted(node):
        if not node:
            return False
        node_data = agent_instance.graph.graph.nodes.get(node.id, {})
        return node_data.get("properties", {}).get("lexical_promoted_as") is not None

    is_definitional = relation_type in {
        "is_a",
        "is_located_in",
        "is_part_of",
        "has_property",
        "has_capital",
        "is_capital_of",
    }
    is_high_trust_source = (
        provenance in ("llm_verified", "dictionary", "seed") or confidence >= 0.85
    )

    if is_definitional and is_high_trust_source:
        if not is_node_promoted(subj_node):
            promote_word(agent_instance, subject_name, "noun_phrase", confidence=0.9)
        if not is_node_promoted(obj_node):
            promote_word(agent_instance, object_name, "noun_phrase", confidence=0.9)

    if not is_node_promoted(subj_node) or not is_node_promoted(obj_node):
        add_pending_relation(agent_instance, relation, interpretation)
        return "deferred"

    new_conf = float(interpretation.get("confidence", 0.6))
    new_neg = bool(interpretation.get("negated", False))
    new_provenance = interpretation.get("provenance", "user")

    new_props = {
        "negated": new_neg,
        "provenance": new_provenance,
        "confidence": new_conf,
    }

    existing_edges = [
        e
        for e in agent_instance.graph.get_edges_from_node(subj_node.id)
        if e.type == relation_type
        and agent_instance.graph.get_node_by_id(e.target)
        and agent_instance.graph.get_node_by_id(e.target).name == object_name
    ]

    if existing_edges:
        e = existing_edges[0]
        existing_conf = float(e.properties.get("confidence", e.weight))
        existing_neg = bool(e.properties.get("negated", False))

        if existing_neg != new_neg:
            if new_conf > existing_conf + 0.2:
                agent_instance.graph.edges[e.source, e.target, e.id].update(new_props)
                return "replaced"
            new_props["contradicted"] = True
            new_props["confidence"] = max(0.05, new_conf * 0.5)
            agent_instance.graph.add_edge(
                subj_node,
                obj_node,
                relation_type,
                properties=new_props,
            )
            return "contradiction_stored"
        merged_conf = max(existing_conf, new_conf)
        new_props["confidence"] = merged_conf
        agent_instance.graph.edges[e.source, e.target, e.id].update(new_props)
        return "inserted"

    agent_instance.graph.add_edge(
        subj_node,
        obj_node,
        relation_type,
        new_conf,
        properties=new_props,
    )
    return "inserted"
