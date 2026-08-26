"""school_math -> short format: keep only the 解答 section, strip markup, verify every a op b = c."""
import json, os, re, sys
sys.path.insert(0,"/work/aupai")
from algorithms.rlvr_reward import extract_boxed, normalize_answer, to_number
SRC="/work/aupai/data/workbatch/school_math_train.jsonl"; DST="/work/aupai/data/workbatch/school_math_short.jsonl"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from eqcheck import iter_equations, OPS  # noqa: E402

def strip(s):
    s=re.sub(r'\*\*(.+?)\*\*',r'\1',s); s=re.sub(r'\\[\(\)\[\]]','',s); s=s.replace('\\left','').replace('\\right','').replace('\\approx','≈').replace('\\leq','≤').replace('\\geq','≥').replace('\\Rightarrow','⇒').replace('\\%','%').replace('\\quad',' ').replace('\\,',' ').replace('\\!','').replace('\\times','×').replace('\\div','÷').replace('\\cdot','×')
    s=re.sub(r'\\[dt]?frac\{([^{}]*)\}\{([^{}]*)\}',r'\1/\2',s); s=re.sub(r'\\(?:text|mathrm)\{([^{}]*)\}',r'\1',s)
    s=re.sub(r'^\s*[-*]\s+','',s,flags=re.M); s=re.sub(r'^[ \t]+','',s,flags=re.M); s=re.sub(r'^\s*\d+\.\s+','',s,flags=re.M)
    s=re.sub(r'[ \t]+',' ',s); s=re.sub(r'\n\s*\n+','\n',s); return s.strip()
def main():
    st=dict(n=0,no_marker=0,multi_box=0,no_box=0,bad_eq=0,long=0,nonnum=0,latex=0,kept=0)
    with open(DST,'w',encoding='utf-8') as out:
        for l in open(SRC,encoding='utf-8'):
            d=json.loads(l); o=d['output']; st['n']+=1
            m=re.search(r'\n\**(?:解答|解题过程|解题步骤|步骤如下|计算过程如下|计算如下)[:：]?\**\s*\n',o)
            if not m: st['no_marker']+=1; continue
            body=strip(o[m.end():])
            if body.count('\\boxed')!=1: st['multi_box' if '\\boxed' in body else 'no_box']+=1; continue
            ok=True
            for a,op,b,c in iter_equations(body):
                r=OPS[op](float(a),float(b))
                if r is None or abs(r-float(c))>1e-6*max(1,abs(r)): ok=False; break
            if not ok: st['bad_eq']+=1; continue
            if to_number(normalize_answer(extract_boxed(body))) is None: st['nonnum']+=1; continue
            q=d['instruction'].strip()
            body='\n'.join(ln for ln in body.split('\n') if ln.strip() and ln.strip() not in q and '？' not in ln)
            if len(body)>600: st['long']+=1; continue
            if '\\' in body.replace('\\boxed',''): st['latex']+=1; continue
            st['kept']+=1; out.write(json.dumps({'instruction':d['instruction'],'output':body},ensure_ascii=False)+'\n')
    print(st)


if __name__ == "__main__":
    main()
