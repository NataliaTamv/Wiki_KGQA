import os, sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)   

import sys

from src import sparql_executor, subgraph


def main():
    if len(sys.argv) < 3:
        raise SystemExit('Χρήση: python debug_subgraph.py Q8027 "celebrated on" [min_score]')
    qid, rel = sys.argv[1], sys.argv[2]
    min_score = float(sys.argv[3]) if len(sys.argv) > 3 else 0.6

    print("0. Έλεγχος endpoint (Q31 P36 -> αναμένεται .../entity/Q239):")
    rows = sparql_executor.run_sparql(
        "SELECT ?o WHERE { <http://www.wikidata.org/entity/Q31> <http://www.wikidata.org/prop/direct/P36> ?o }"
    )
    print("   ", rows if rows else "ΚΕΝΟ ΑΠΟΤΕΛΕΣΜΑ")

    print(f"\n1. Properties του {qid} στο subgraph:")
    props = subgraph.fetch_subgraph(qid)
    print(f"    {len(props)} properties")
    if not props:
        print("    -> ΚΕΝΟ: το query του subgraph δεν επέστρεψε properties.")
        return

    print(f'\n2. Κατάταξη για το relation "{rel}" (όριο min_score = {min_score}):')
    ranked = subgraph.rank_properties(rel, props, top_k=10)
    for r in ranked:
        mark = "ΠΕΡΝΑΕΙ" if r["score"] >= min_score else "      "
        print(f"    {r['score']:.3f} {mark}  {r['property_id']:>7} {r['direction']:>3}  {r['property_label']}")
    best = ranked[0]["score"] if ranked else 0.0
    if best < min_score:
        print(f"\n    -> Κανένα property δεν φτάνει το {min_score} (καλύτερο: {best:.3f}). "
              f"Το _walk_chain επιστρέφει None ΣΙΩΠΗΛΑ, και το pipeline πέφτει στο LLM.")
    else:
        print("\n    -> Υπάρχει property που περνά το όριο, άρα το subgraph θα έβρισκε μονοπάτι για αυτό το entity.")


if __name__ == "__main__":
    main()