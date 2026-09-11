#!/usr/bin/env python3
"""Print the gate-relevant fields of a domain's build_corpus_stats.json.

Usage: read_aggregate_stats.py <final_domain_dir>
"""
import json
import sys


def main():
    d = sys.argv[1]
    s = json.load(open(f"{d}/build_corpus_stats.json"))
    print(f"domain {s.get('domain')}")
    print(f"kept_tokens {s['kept_tokens']} ({s['kept_tokens']/1e9:.4f} B)")
    print(f"kept {s['kept']} shards {s['n_shards']}")
    ng = s.get("decontam_ngram", {})
    print(f"ngram13_in {ng.get('rows_in')} dropped {ng.get('rows_dropped')} "
          f"problems {ng.get('distinct_problems_hit')}")
    print(f"cross_group_dup {s['reasons'].get('cross_group_dup')} "
          f"nontriv_cut {s['reasons'].get('non_trivial')}")
    print(f"fingerprint {s['fingerprint']}")
    print(f"decontam_fp {s.get('decontam_fp')}")


if __name__ == "__main__":
    main()
