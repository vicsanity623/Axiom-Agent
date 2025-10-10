from __future__ import annotations

# knowledge_base.py
from typing import TYPE_CHECKING

from axiom.dictionary_utils import get_word_info_from_wordnet

if TYPE_CHECKING:
    from axiom.cognitive_agent import CognitiveAgent


def seed_domain_knowledge(agent_instance: CognitiveAgent) -> None:
    """Seed the agent's brain with a large, foundational set of facts.

    This function populates a new knowledge graph with a wide range of
    pre-defined knowledge, giving the agent a significant head start.
    It is called only when the agent detects it is starting with a fresh,
    empty brain.

    The seeded knowledge includes:
    - The agent's own identity and purpose.
    - Foundational world knowledge (physics, geography, biology).
    - Abstract concepts (colors, sentiments).
    - Integration with WordNet to enrich seeded concepts with hypernyms.

    Args:
        agent_instance: The instance of the CognitiveAgent to be seeded.
    """
    print("   - Seeding a vast initial world knowledge base...")

    print("     - Seeding self-identity...")
    agent_instance.manual_add_knowledge(
        "agent",
        "concept",
        "has_name",
        "Axiom",
        weight=1.0,
    )
    agent_instance.manual_add_knowledge(
        "agent",
        "concept",
        "is_a",
        "cognitive agent",
        weight=1.0,
    )
    agent_instance.manual_add_knowledge(
        "cognitive agent",
        "concept",
        "is_a",
        "AI assistant",
        weight=0.9,
    )

    agent_instance.manual_add_knowledge(
        "agent",
        "concept",
        "has_ability",
        "learn",
        weight=0.9,
    )
    agent_instance.manual_add_knowledge(
        "agent",
        "concept",
        "has_ability",
        "reason",
        weight=0.8,
    )
    agent_instance.manual_add_knowledge(
        "agent",
        "concept",
        "has_ability",
        "understand",
        weight=0.8,
    )
    agent_instance.manual_add_knowledge(
        "agent",
        "concept",
        "has_purpose",
        "assist users",
        weight=0.9,
    )
    agent_instance.manual_add_knowledge(
        "agent",
        "concept",
        "communicates_via",
        "text",
        weight=1.0,
    )
    agent_instance.manual_add_knowledge(
        "agent",
        "concept",
        "exists_on",
        "a server",
        weight=1.0,
    )
    agent_instance.manual_add_knowledge(
        "agent",
        "concept",
        "has_ability",
        "problem solving",
        weight=0.8,
    )
    agent_instance.manual_add_knowledge(
        "agent",
        "concept",
        "was_created_by",
        "humans",
        weight=0.9,
    )
    agent_instance.manual_add_knowledge(
        "agent",
        "concept",
        "operates_using",
        "algorithms",
        weight=0.8,
    )
    agent_instance.manual_add_knowledge(
        "agent",
        "concept",
        "has_ability",
        "improve over time",
        weight=0.7,
    )
    agent_instance.manual_add_knowledge(
        "agent",
        "concept",
        "requires",
        "computation",
        weight=0.9,
    )
    agent_instance.manual_add_knowledge(
        "agent",
        "concept",
        "has_capability",
        "natural language processing",
        weight=0.8,
    )
    agent_instance.manual_add_knowledge(
        "agent",
        "concept",
        "has_ability",
        "follow instructions",
        weight=0.9,
    )
    agent_instance.manual_add_knowledge(
        "agent",
        "concept",
        "has_ability",
        "access knowledge",
        weight=0.9,
    )
    agent_instance.manual_add_knowledge(
        "agent",
        "concept",
        "has_limitation",
        "no physical body",
        weight=0.8,
    )
    agent_instance.manual_add_knowledge(
        "agent",
        "concept",
        "functions_through",
        "code",
        weight=0.9,
    )
    agent_instance.manual_add_knowledge(
        "agent",
        "concept",
        "has_property",
        "long-term memory",
        weight=0.8,
    )
    agent_instance.manual_add_knowledge(
        "agent",
        "concept",
        "has_ability",
        "process information",
        weight=0.9,
    )
    agent_instance.manual_add_knowledge(
        "agent",
        "concept",
        "has_ability",
        "respond to queries",
        weight=0.9,
    )
    agent_instance.manual_add_knowledge(
        "agent",
        "concept",
        "operates_on",
        "electricity",
        weight=0.8,
    )
    agent_instance.manual_add_knowledge(
        "agent",
        "concept",
        "has_goal",
        "to be helpful",
        weight=0.8,
    )
    agent_instance.manual_add_knowledge(
        "agent",
        "concept",
        "has_ability",
        "explain concepts",
        weight=0.7,
    )
    agent_instance.manual_add_knowledge(
        "agent",
        "concept",
        "has_ability",
        "understand context",
        weight=0.7,
    )
    agent_instance.manual_add_knowledge(
        "agent",
        "concept",
        "has_ability",
        "summarize information",
        weight=0.7,
    )
    agent_instance.manual_add_knowledge(
        "agent",
        "concept",
        "has_limitation",
        "no emotions",
        weight=0.8,
    )
    agent_instance.manual_add_knowledge(
        "agent",
        "concept",
        "has_ability",
        "translate languages",
        weight=0.7,
    )
    agent_instance.manual_add_knowledge(
        "agent",
        "concept",
        "uses",
        "pattern recognition",
        weight=0.8,
    )
    agent_instance.manual_add_knowledge(
        "agent",
        "concept",
        "has_ability",
        "generate creative content",
        weight=0.7,
    )
    agent_instance.manual_add_knowledge(
        "agent",
        "concept",
        "has_property",
        "consistency in responses",
        weight=0.7,
    )
    agent_instance.manual_add_knowledge(
        "agent",
        "concept",
        "has_ability",
        "analyze data",
        weight=0.8,
    )
    agent_instance.manual_add_knowledge(
        "agent",
        "concept",
        "has_ability",
        "operate 24/7",
        weight=0.9,
    )
    agent_instance.manual_add_knowledge(
        "agent",
        "concept",
        "has_access_to",
        "vast information",
        weight=0.8,
    )
    agent_instance.manual_add_knowledge(
        "agent",
        "concept",
        "has_ability",
        "adapt to new topics",
        weight=0.7,
    )
    agent_instance.manual_add_knowledge(
        "agent",
        "concept",
        "has_ability",
        "follow ethical guidelines",
        weight=0.8,
    )
    agent_instance.manual_add_knowledge(
        "agent",
        "concept",
        "has_ability",
        "collaborate with humans",
        weight=0.7,
    )
    agent_instance.manual_add_knowledge(
        "agent",
        "concept",
        "has_property",
        "potential for continuous improvement",
        weight=0.7,
    )

    print("     - Seeding world knowledge...")
    agent_instance.manual_add_knowledge(
        "sky",
        "concept",
        "has_property",
        "blue",
        weight=0.9,
    )
    agent_instance.manual_add_knowledge("sun", "concept", "is_a", "star", weight=1.0)
    agent_instance.manual_add_knowledge("sun", "concept", "emits", "light", weight=1.0)
    agent_instance.manual_add_knowledge(
        "water",
        "concept",
        "is_a",
        "liquid",
        weight=0.8,
    )
    agent_instance.manual_add_knowledge(
        "fire",
        "concept",
        "has_property",
        "hot",
        weight=1.0,
    )
    agent_instance.manual_add_knowledge(
        "moon",
        "concept",
        "reflects",
        "light",
        weight=0.9,
    )
    agent_instance.manual_add_knowledge(
        "gravity",
        "concept",
        "affects",
        "all objects",
        weight=0.9,
    )
    agent_instance.manual_add_knowledge(
        "rain",
        "concept",
        "is_composed_of",
        "water droplets",
        weight=0.9,
    )
    agent_instance.manual_add_knowledge(
        "wind",
        "concept",
        "is_a",
        "moving air",
        weight=0.9,
    )
    agent_instance.manual_add_knowledge(
        "ice",
        "concept",
        "is_a",
        "frozen water",
        weight=0.9,
    )
    agent_instance.manual_add_knowledge(
        "cloud",
        "concept",
        "is_made_of",
        "water vapor",
        weight=0.9,
    )
    agent_instance.manual_add_knowledge(
        "earth",
        "planet",
        "has_property",
        "gravity",
        weight=0.9,
    )
    agent_instance.manual_add_knowledge(
        "sound",
        "concept",
        "travels_through",
        "air",
        weight=0.8,
    )
    agent_instance.manual_add_knowledge(
        "light",
        "concept",
        "is_faster_than",
        "sound",
        weight=0.9,
    )
    agent_instance.manual_add_knowledge(
        "metal",
        "concept",
        "conducts",
        "electricity",
        weight=0.9,
    )
    agent_instance.manual_add_knowledge(
        "magnet",
        "concept",
        "attracts",
        "iron",
        weight=0.9,
    )
    agent_instance.manual_add_knowledge(
        "volcano",
        "concept",
        "erupts",
        "lava",
        weight=0.9,
    )
    agent_instance.manual_add_knowledge(
        "earthquake",
        "concept",
        "causes",
        "ground shaking",
        weight=0.9,
    )
    agent_instance.manual_add_knowledge(
        "tornado",
        "concept",
        "is_a",
        "violent storm",
        weight=0.9,
    )
    agent_instance.manual_add_knowledge(
        "atom",
        "concept",
        "is_a",
        "smallest unit of matter",
        weight=0.9,
    )
    agent_instance.manual_add_knowledge(
        "energy",
        "concept",
        "has_property",
        "cannot be created or destroyed",
        weight=0.8,
    )
    agent_instance.manual_add_knowledge(
        "friction",
        "concept",
        "causes",
        "heat",
        weight=0.8,
    )
    agent_instance.manual_add_knowledge(
        "vacuum",
        "concept",
        "has_property",
        "no air",
        weight=0.9,
    )
    agent_instance.manual_add_knowledge(
        "lightning",
        "concept",
        "is_a",
        "electrical discharge",
        weight=0.9,
    )
    agent_instance.manual_add_knowledge(
        "echo",
        "concept",
        "is_a",
        "reflected sound",
        weight=0.9,
    )
    agent_instance.manual_add_knowledge(
        "lens",
        "concept",
        "refracts",
        "light",
        weight=0.8,
    )
    agent_instance.manual_add_knowledge(
        "prism",
        "concept",
        "splits",
        "white light",
        weight=0.8,
    )
    agent_instance.manual_add_knowledge(
        "evaporation",
        "concept",
        "changes",
        "liquid to gas",
        weight=0.9,
    )
    agent_instance.manual_add_knowledge(
        "condensation",
        "concept",
        "changes",
        "gas to liquid",
        weight=0.9,
    )
    agent_instance.manual_add_knowledge(
        "orbit",
        "concept",
        "is_a",
        "path around a planet",
        weight=0.9,
    )
    agent_instance.manual_add_knowledge(
        "comet",
        "concept",
        "has_property",
        "a tail",
        weight=0.9,
    )
    agent_instance.manual_add_knowledge(
        "galaxy",
        "concept",
        "contains",
        "stars",
        weight=0.9,
    )
    agent_instance.manual_add_knowledge(
        "black hole",
        "concept",
        "has_property",
        "strong gravity",
        weight=0.8,
    )
    agent_instance.manual_add_knowledge(
        "supernova",
        "concept",
        "is_a",
        "exploding star",
        weight=0.8,
    )
    agent_instance.manual_add_knowledge(
        "nebula",
        "concept",
        "is_a",
        "cloud of gas and dust",
        weight=0.8,
    )

    agent_instance.manual_add_knowledge(
        "earth",
        "planet",
        "has_satellite",
        "the moon",
        weight=1.0,
    )
    agent_instance.manual_add_knowledge(
        "paris",
        "city",
        "is_located_in",
        "france",
        weight=1.0,
    )
    agent_instance.manual_add_knowledge(
        "france",
        "country",
        "is_located_in",
        "europe",
        weight=1.0,
    )
    agent_instance.manual_add_knowledge(
        "mount everest",
        "mountain",
        "is_a",
        "tallest mountain",
        weight=1.0,
    )
    agent_instance.manual_add_knowledge(
        "london",
        "city",
        "is_capital_of",
        "england",
        weight=1.0,
    )
    agent_instance.manual_add_knowledge(
        "nile",
        "river",
        "is_located_in",
        "africa",
        weight=1.0,
    )
    agent_instance.manual_add_knowledge(
        "sahara",
        "desert",
        "is_located_in",
        "africa",
        weight=1.0,
    )
    agent_instance.manual_add_knowledge(
        "pacific",
        "ocean",
        "is_a",
        "largest ocean",
        weight=1.0,
    )
    agent_instance.manual_add_knowledge(
        "asia",
        "continent",
        "has_property",
        "highest population",
        weight=0.9,
    )
    agent_instance.manual_add_knowledge(
        "amazon",
        "river",
        "is_located_in",
        "south america",
        weight=1.0,
    )
    agent_instance.manual_add_knowledge(
        "new york",
        "city",
        "is_located_in",
        "united states",
        weight=1.0,
    )
    agent_instance.manual_add_knowledge(
        "tokyo",
        "city",
        "is_capital_of",
        "japan",
        weight=1.0,
    )
    agent_instance.manual_add_knowledge(
        "antarctica",
        "continent",
        "is_a",
        "coldest continent",
        weight=1.0,
    )
    agent_instance.manual_add_knowledge(
        "australia",
        "country",
        "is_a",
        "continent",
        weight=1.0,
    )
    agent_instance.manual_add_knowledge(
        "rome",
        "city",
        "is_capital_of",
        "italy",
        weight=1.0,
    )
    agent_instance.manual_add_knowledge(
        "ganges",
        "river",
        "is_located_in",
        "india",
        weight=1.0,
    )
    agent_instance.manual_add_knowledge(
        "himalayas",
        "mountain range",
        "is_located_in",
        "asia",
        weight=1.0,
    )
    agent_instance.manual_add_knowledge(
        "dead sea",
        "lake",
        "is_a",
        "lowest point on land",
        weight=1.0,
    )
    agent_instance.manual_add_knowledge(
        "greenland",
        "island",
        "is_a",
        "largest island",
        weight=1.0,
    )
    agent_instance.manual_add_knowledge(
        "madagascar",
        "island",
        "is_located_off",
        "africa",
        weight=1.0,
    )
    agent_instance.manual_add_knowledge(
        "siberia",
        "region",
        "is_located_in",
        "russia",
        weight=1.0,
    )
    agent_instance.manual_add_knowledge(
        "amazon rainforest",
        "forest",
        "is_located_in",
        "south america",
        weight=1.0,
    )
    agent_instance.manual_add_knowledge(
        "grand canyon",
        "canyon",
        "is_located_in",
        "arizona",
        weight=1.0,
    )
    agent_instance.manual_add_knowledge(
        "great barrier reef",
        "reef",
        "is_located_off",
        "australia",
        weight=1.0,
    )
    agent_instance.manual_add_knowledge(
        "yellowstone",
        "national park",
        "has_property",
        "geysers",
        weight=1.0,
    )
    agent_instance.manual_add_knowledge(
        "niagara falls",
        "waterfall",
        "is_located_on_border_of",
        "usa and canada",
        weight=1.0,
    )
    agent_instance.manual_add_knowledge(
        "mississippi",
        "river",
        "is_a",
        "longest river in usa",
        weight=1.0,
    )
    agent_instance.manual_add_knowledge(
        "alps",
        "mountain range",
        "is_located_in",
        "europe",
        weight=1.0,
    )
    agent_instance.manual_add_knowledge(
        "red sea",
        "sea",
        "is_located_between",
        "africa and asia",
        weight=1.0,
    )
    agent_instance.manual_add_knowledge(
        "caspian sea",
        "body of water",
        "is_a",
        "largest lake",
        weight=1.0,
    )
    agent_instance.manual_add_knowledge(
        "andes",
        "mountain range",
        "is_located_in",
        "south america",
        weight=1.0,
    )
    agent_instance.manual_add_knowledge(
        "kilimanjaro",
        "mountain",
        "is_located_in",
        "tanzania",
        weight=1.0,
    )
    agent_instance.manual_add_knowledge(
        "gobi",
        "desert",
        "is_located_in",
        "asia",
        weight=1.0,
    )

    agent_instance.manual_add_knowledge(
        "human",
        "species",
        "is_a",
        "mammal",
        weight=1.0,
    )
    agent_instance.manual_add_knowledge("dog", "species", "is_a", "mammal", weight=1.0)
    agent_instance.manual_add_knowledge("cat", "species", "is_a", "mammal", weight=1.0)
    agent_instance.manual_add_knowledge("mammal", "class", "is_a", "animal", weight=1.0)
    agent_instance.manual_add_knowledge(
        "animal",
        "kingdom",
        "is_a",
        "living thing",
        weight=1.0,
    )
    agent_instance.manual_add_knowledge(
        "tree",
        "plant",
        "is_a",
        "living thing",
        weight=1.0,
    )
    agent_instance.manual_add_knowledge(
        "fish",
        "animal",
        "lives_in",
        "water",
        weight=0.9,
    )
    agent_instance.manual_add_knowledge(
        "bird",
        "animal",
        "has_property",
        "feathers",
        weight=0.9,
    )
    agent_instance.manual_add_knowledge(
        "insect",
        "animal",
        "has_property",
        "six legs",
        weight=0.9,
    )
    agent_instance.manual_add_knowledge(
        "reptile",
        "animal",
        "is_a",
        "cold-blooded",
        weight=0.9,
    )
    agent_instance.manual_add_knowledge(
        "plant",
        "living thing",
        "produces",
        "oxygen",
        weight=0.9,
    )
    agent_instance.manual_add_knowledge(
        "mammal",
        "animal",
        "has_ability",
        "feed milk to young",
        weight=0.9,
    )
    agent_instance.manual_add_knowledge(
        "human",
        "mammal",
        "has_property",
        "opposable thumbs",
        weight=0.9,
    )
    agent_instance.manual_add_knowledge(
        "dolphin",
        "mammal",
        "lives_in",
        "ocean",
        weight=0.9,
    )
    agent_instance.manual_add_knowledge(
        "bat",
        "mammal",
        "has_ability",
        "fly",
        weight=0.9,
    )
    agent_instance.manual_add_knowledge(
        "amphibian",
        "animal",
        "lives_in",
        "land and water",
        weight=0.9,
    )
    agent_instance.manual_add_knowledge(
        "fungus",
        "living thing",
        "is_not_a",
        "plant or animal",
        weight=0.8,
    )
    agent_instance.manual_add_knowledge(
        "whale",
        "mammal",
        "is_a",
        "largest animal",
        weight=0.9,
    )
    agent_instance.manual_add_knowledge(
        "bee",
        "insect",
        "produces",
        "honey",
        weight=0.9,
    )
    agent_instance.manual_add_knowledge(
        "spider",
        "arachnid",
        "has_property",
        "eight legs",
        weight=0.9,
    )
    agent_instance.manual_add_knowledge(
        "butterfly",
        "insect",
        "undergoes",
        "metamorphosis",
        weight=0.9,
    )
    agent_instance.manual_add_knowledge(
        "cactus",
        "plant",
        "stores",
        "water",
        weight=0.9,
    )
    agent_instance.manual_add_knowledge(
        "venus flytrap",
        "plant",
        "eats",
        "insects",
        weight=0.9,
    )
    agent_instance.manual_add_knowledge(
        "coral",
        "animal",
        "builds",
        "reefs",
        weight=0.9,
    )
    agent_instance.manual_add_knowledge(
        "octopus",
        "animal",
        "has_property",
        "eight arms",
        weight=0.9,
    )
    agent_instance.manual_add_knowledge(
        "kangaroo",
        "marsupial",
        "has_property",
        "pouch",
        weight=0.9,
    )
    agent_instance.manual_add_knowledge("penguin", "bird", "cannot", "fly", weight=0.9)
    agent_instance.manual_add_knowledge(
        "elephant",
        "mammal",
        "has_property",
        "trunk",
        weight=0.9,
    )
    agent_instance.manual_add_knowledge(
        "giraffe",
        "mammal",
        "has_property",
        "long neck",
        weight=0.9,
    )
    agent_instance.manual_add_knowledge(
        "shark",
        "fish",
        "has_property",
        "cartilage skeleton",
        weight=0.9,
    )
    agent_instance.manual_add_knowledge(
        "snake",
        "reptile",
        "has_property",
        "no legs",
        weight=0.9,
    )
    agent_instance.manual_add_knowledge(
        "turtle",
        "reptile",
        "has_property",
        "shell",
        weight=0.9,
    )
    agent_instance.manual_add_knowledge(
        "frog",
        "amphibian",
        "undergoes",
        "metamorphosis",
        weight=0.9,
    )
    agent_instance.manual_add_knowledge(
        "algae",
        "organism",
        "lives_in",
        "water",
        weight=0.9,
    )
    agent_instance.manual_add_knowledge(
        "mushroom",
        "fungus",
        "reproduces_with",
        "spores",
        weight=0.9,
    )
    agent_instance.manual_add_knowledge(
        "bacteria",
        "microorganism",
        "is_a",
        "single-celled organism",
        weight=0.9,
    )
    agent_instance.manual_add_knowledge(
        "virus",
        "microorganism",
        "requires",
        "host to reproduce",
        weight=0.9,
    )

    agent_instance.manual_add_knowledge("apple", "fruit", "is_a", "food", weight=0.9)
    agent_instance.manual_add_knowledge("banana", "fruit", "is_a", "food", weight=0.9)
    agent_instance.manual_add_knowledge(
        "carrot",
        "vegetable",
        "is_a",
        "food",
        weight=0.9,
    )
    agent_instance.manual_add_knowledge(
        "bread",
        "food",
        "is_made_from",
        "flour",
        weight=0.9,
    )
    agent_instance.manual_add_knowledge(
        "cheese",
        "food",
        "is_made_from",
        "milk",
        weight=0.9,
    )
    agent_instance.manual_add_knowledge("rice", "food", "is_a", "grain", weight=0.9)
    agent_instance.manual_add_knowledge("chicken", "food", "is_a", "meat", weight=0.9)
    agent_instance.manual_add_knowledge(
        "chocolate",
        "food",
        "is_made_from",
        "cocoa",
        weight=0.9,
    )
    agent_instance.manual_add_knowledge(
        "pasta",
        "food",
        "is_made_from",
        "wheat",
        weight=0.9,
    )
    agent_instance.manual_add_knowledge("soup", "food", "is_a", "liquid", weight=0.8)
    agent_instance.manual_add_knowledge(
        "salad",
        "food",
        "contains",
        "vegetables",
        weight=0.9,
    )
    agent_instance.manual_add_knowledge(
        "ice cream",
        "food",
        "has_property",
        "cold",
        weight=0.9,
    )
    agent_instance.manual_add_knowledge(
        "pizza",
        "food",
        "has_component",
        "crust",
        weight=0.9,
    )
    agent_instance.manual_add_knowledge(
        "orange",
        "fruit",
        "is_a",
        "citrus fruit",
        weight=0.9,
    )
    agent_instance.manual_add_knowledge(
        "potato",
        "vegetable",
        "is_a",
        "root vegetable",
        weight=0.9,
    )
    agent_instance.manual_add_knowledge(
        "tomato",
        "fruit",
        "is_used_as",
        "a vegetable",
        weight=0.9,
    )
    agent_instance.manual_add_knowledge(
        "onion",
        "vegetable",
        "has_property",
        "layers",
        weight=0.9,
    )
    agent_instance.manual_add_knowledge(
        "garlic",
        "vegetable",
        "is_used_for",
        "flavoring",
        weight=0.9,
    )
    agent_instance.manual_add_knowledge(
        "beef",
        "meat",
        "comes_from",
        "cows",
        weight=0.9,
    )
    agent_instance.manual_add_knowledge(
        "pork",
        "meat",
        "comes_from",
        "pigs",
        weight=0.9,
    )
    agent_instance.manual_add_knowledge(
        "fish",
        "food",
        "is_a",
        "source of protein",
        weight=0.9,
    )
    agent_instance.manual_add_knowledge(
        "egg",
        "food",
        "comes_from",
        "chickens",
        weight=0.9,
    )
    agent_instance.manual_add_knowledge(
        "milk",
        "beverage",
        "comes_from",
        "cows",
        weight=0.9,
    )
    agent_instance.manual_add_knowledge(
        "yogurt",
        "food",
        "is_made_from",
        "milk",
        weight=0.9,
    )
    agent_instance.manual_add_knowledge(
        "honey",
        "food",
        "is_made_by",
        "bees",
        weight=0.9,
    )
    agent_instance.manual_add_knowledge(
        "coffee",
        "beverage",
        "is_made_from",
        "beans",
        weight=0.9,
    )
    agent_instance.manual_add_knowledge(
        "tea",
        "beverage",
        "is_made_from",
        "leaves",
        weight=0.9,
    )
    agent_instance.manual_add_knowledge(
        "sugar",
        "ingredient",
        "has_property",
        "sweet",
        weight=0.9,
    )
    agent_instance.manual_add_knowledge(
        "salt",
        "ingredient",
        "has_property",
        "salty",
        weight=0.9,
    )
    agent_instance.manual_add_knowledge(
        "flour",
        "ingredient",
        "is_made_from",
        "grains",
        weight=0.9,
    )
    agent_instance.manual_add_knowledge(
        "butter",
        "dairy",
        "is_made_from",
        "milk",
        weight=0.9,
    )
    agent_instance.manual_add_knowledge(
        "oil",
        "ingredient",
        "is_used_for",
        "cooking",
        weight=0.9,
    )
    agent_instance.manual_add_knowledge(
        "vinegar",
        "ingredient",
        "has_property",
        "sour",
        weight=0.9,
    )

    print("     - Seeding abstract concepts...")
    color_node = agent_instance._add_or_update_concept("color", "attribute")
    colors = [
        "red",
        "green",
        "yellow",
        "blue",
        "orange",
        "purple",
        "black",
        "white",
        "brown",
        "pink",
        "gray",
        "cyan",
        "magenta",
        "gold",
        "silver",
        "beige",
        "turquoise",
        "lavender",
        "maroon",
        "navy",
        "olive",
        "teal",
        "coral",
        "indigo",
        "violet",
        "crimson",
        "khaki",
        "plum",
        "salmon",
        "tan",
        "mint",
    ]
    for color_name in colors:
        node = agent_instance._add_or_update_concept(color_name, "descriptor")
        if node and color_node:
            agent_instance.graph.add_edge(node, color_node, "is_a", 0.9)

    sentiment_node = agent_instance._add_or_update_concept("sentiment", "attribute")
    sentiments = [
        "happy",
        "sad",
        "angry",
        "excited",
        "fearful",
        "surprised",
        "disgusted",
        "calm",
        "confused",
        "proud",
        "jealous",
        "anxious",
        "content",
        "curious",
        "depressed",
        "embarrassed",
        "enthusiastic",
        "frustrated",
        "grateful",
        "guilty",
        "hopeful",
        "impatient",
        "inspired",
        "lonely",
        "nostalgic",
        "optimistic",
        "pessimistic",
        "relieved",
        "romantic",
        "satisfied",
        "sympathetic",
    ]
    for sentiment_name in sentiments:
        node = agent_instance._add_or_update_concept(sentiment_name, "descriptor")
        if node and sentiment_node:
            agent_instance.graph.add_edge(node, sentiment_node, "is_a", 0.8)

    print("     - Integrating WordNet definitions for seeded concepts...")

    seeded_words = {
        data["name"] for _, data in agent_instance.graph.graph.nodes(data=True)
    }

    for word in list(seeded_words):
        if len(word.split()) > 1:
            continue

        word_info = get_word_info_from_wordnet(word)
        if word_info["hypernyms_raw"]:
            main_node = agent_instance.graph.get_node_by_name(word)

            for hypernym_word in word_info["hypernyms_raw"][:1]:
                hypernym_node = agent_instance._add_or_update_concept(hypernym_word)
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

    for word, pos in core_vocab.items():
        agent_instance.lexicon.add_linguistic_knowledge(word, pos)
