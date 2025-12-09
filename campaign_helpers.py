# src/campaign_helpers.py
from typing import Dict

def suggest_campaign(cluster_kpi: Dict) -> Dict:
    """
    Return a simple campaign suggestion dict for a cluster using its KPIs.
    Expected keys in cluster_kpi: 'avg_monetary', 'avg_frequency', 'avg_sentiment', 'total_revenue'
    """
    avg_m = float(cluster_kpi.get("avg_monetary", 0))
    avg_f = float(cluster_kpi.get("avg_frequency", 0))
    avg_s = float(cluster_kpi.get("avg_sentiment", 0))

    if avg_s < -0.05 and avg_m > 100:
        return {
            "title": "VIP Recovery Pack",
            "offer": "20% off + priority support",
            "est_uplift": 0.08,
            "notes": "Prioritize human follow-up; sentiment negative + high value"
        }
    if avg_f < 1:
        return {
            "title": "Win-Back Offer",
            "offer": "10% off + free shipping",
            "est_uplift": 0.05,
            "notes": "Low frequency customers — incentive to return"
        }
    return {
        "title": "Cross-Sell Nudge",
        "offer": "Recommended items + 5% off",
        "est_uplift": 0.03,
        "notes": "General cross-sell to increase AOV"
    }
