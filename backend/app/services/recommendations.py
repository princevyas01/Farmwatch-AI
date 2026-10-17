"""Curated, low-risk agronomic recommendations mapping.

STRICT CONSTRAINTS:
1. No specific chemical pesticide or fungicide dosage prescriptions.
2. Focus on low-risk cultural, monitoring, sanitation, and physical management practices.
3. Explicitly advise agronomist consultation for commercial or uncertain situations.
4. Avoid categorical destruction advice based on automated statistical predictions.
5. Do NOT treat general leaf classes (e.g., 'Raspberry leaf', 'Peach leaf') as certified 'healthy'.
"""

from typing import List, Dict

# Explicit class-to-recommendations mapping for the 11 audited classes
EXPLICIT_CLASS_RECOMMENDATIONS: Dict[str, List[str]] = {
    # 1. Blueberry leaf (General foliage category - not a certified healthy diagnosis)
    "Blueberry leaf": [
        "Dataset Category: General blueberry foliage sample without specific disease annotation.",
        "Maintain routine scouting and monitor lower and upper leaf surfaces for discoloration.",
        "Ensure soil pH remains in the optimal acidic range (4.5–5.5) for blueberry nutrient uptake.",
        "Consult a local agricultural extension officer if unusual spot or chlorosis symptoms develop.",
    ],
    # 2. Corn leaf blight
    "Corn leaf blight": [
        "Improve field drainage and avoid overhead sprinkler irrigation that lengthens leaf wetness.",
        "Ensure adequate plant spacing to facilitate airflow and rapid morning canopy drying.",
        "Scout field margins for spread of elliptical grayish-green or tan lesions.",
        "Consult an agricultural extension specialist regarding disease-resistant hybrids for subsequent rotations.",
    ],
    # 3. Corn rust leaf
    "Corn rust leaf": [
        "Monitor upper leaf surfaces for reddish-brown to cinnamon pustules (common/southern rust signs).",
        "Avoid excessive early-season nitrogen applications that promote overly succulent, susceptible tissue.",
        "Maintain clean field borders to manage potential weed hosts.",
        "Consult a certified crop advisor to assess economic threshold levels before considering treatments.",
    ],
    # 4. Peach leaf (General foliage category - not a certified healthy diagnosis)
    "Peach leaf": [
        "Dataset Category: General peach foliage sample without specific disease annotation.",
        "Perform routine orchard scouting for leaf curl, shot-hole, or fungal spot symptoms.",
        "Ensure balanced tree canopy pruning during dormant seasons to maximize sunlight penetration.",
        "Consult a fruit tree specialist or local extension office if foliar lesions or deformities appear.",
    ],
    # 5. Potato leaf early blight
    "Potato leaf early blight": [
        "Inspect lower leaves for dark, concentric ringed spots ('target board' pattern).",
        "Remove and dispose of heavily infected lower leaves away from the plot; avoid composting.",
        "Mulch around hill bases to reduce soil splashing onto lower foliage during rainfall.",
        "Practice at least a 3-year crop rotation away from solanaceous plants (potato, tomato, eggplant).",
    ],
    # 6. Potato leaf late blight
    "Potato leaf late blight": [
        "High-priority monitoring: Late blight spreads rapidly under cool, wet, humid conditions.",
        "Inspect foliage for water-soaked, dark lesions with white fungal-like growth on leaf undersides.",
        "Eliminate cull potato piles and volunteer plants that serve as initial inoculum sources.",
        "Promptly submit a leaf sample to an agricultural extension lab for definitive identification.",
    ],
    # 7. Raspberry leaf (General foliage category - not a certified healthy diagnosis)
    "Raspberry leaf": [
        "Dataset Category: General raspberry foliage sample without specific disease annotation.",
        "Prune out old fruiting canes after harvest to maintain good aeration throughout the trellis.",
        "Water at ground level using drip irrigation to keep cane foliage dry.",
        "Consult a small-fruit specialist if yellowing, rust, or viral mosaic patterns are observed.",
    ],
    # 8. Squash Powdery mildew leaf
    "Squash Powdery mildew leaf": [
        "Examine leaves for white, talcum-powder-like fungal patches on upper surfaces.",
        "Space plants generously to promote continuous air movement through the vine canopy.",
        "Prune overhanging vegetation to maximize full daily sun exposure across the bed.",
        "Consult extension guidance regarding approved cultural or biorational preventative options.",
    ],
    # 9. Tomato Septoria leaf spot
    "Tomato Septoria leaf spot": [
        "Inspect lower leaves for small circular spots with dark brown margins and gray centers.",
        "Remove infected lower leaves touching or close to soil level to minimize rain splash transfer.",
        "Apply organic straw or plastic mulch beneath plants to create a barrier against soil-borne spores.",
        "Sanitize stakes, cages, and pruning shears thoroughly between plants.",
    ],
    # 10. Tomato leaf bacterial spot
    "Tomato leaf bacterial spot": [
        "Note: Bacterial spots appear as small, greasy, water-soaked dark spots on leaves and fruit.",
        "Avoid working in or harvesting from tomato rows when leaves are wet to prevent mechanical spread.",
        "Use drip irrigation exclusively; overhead water droplets rapidly transmit bacterial cells.",
        "Use certified disease-free seed and rotate out of solanaceous crops for at least two seasons.",
    ],
    # 11. Tomato leaf late blight
    "Tomato leaf late blight": [
        "High-priority monitoring: Rapidly developing water-soaked lesions with pale borders.",
        "Check lower leaf surfaces during morning hours for white cottony sporulation.",
        "Ensure excellent airflow and avoid handling wet foliage.",
        "Bring a sealed sample to your local extension diagnostic clinic for definitive confirmation.",
    ],
}

DEFAULT_RECOMMENDATIONS = [
    "Monitor nearby plants closely for early signs of spreading symptoms.",
    "Sanitize all pruning tools and equipment with 70% isopropyl alcohol or bleach solution between plants.",
    "Improve air circulation and avoid overhead irrigation to keep leaf surfaces dry.",
    "Consult a certified local agricultural extension officer or plant pathologist before applying chemical treatments.",
]


def get_recommendations_for_class(class_name: str, is_uncertain: bool = False) -> List[str]:
    """Return tailored, safe, low-risk cultural guidance for a predicted leaf condition."""
    if is_uncertain:
        return [
            "Model score is below the heuristic low-score threshold (< 60%). Prediction may be unreliable.",
            "Inspect the whole plant in daylight and examine both upper and lower leaf surfaces.",
            "Take multiple clear, close-up photos of unaffected and affected leaves for comparison.",
            "Bring a physical sample sealed in a plastic bag to a local extension agent or certified crop advisor for confirmation.",
        ]

    # Explicit match
    if class_name in EXPLICIT_CLASS_RECOMMENDATIONS:
        return EXPLICIT_CLASS_RECOMMENDATIONS[class_name]

    # Normalized match fallback
    for key, recs in EXPLICIT_CLASS_RECOMMENDATIONS.items():
        if key.lower() == class_name.lower():
            return recs

    return DEFAULT_RECOMMENDATIONS
