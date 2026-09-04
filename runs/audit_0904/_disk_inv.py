#!/usr/bin/env python3
import os, json
def stats(path):
    nf=0; bs=0; mt=0
    for r, ds, fs in os.walk(path):
        for f in fs:
            p = os.path.join(r, f)
            try:
                if os.path.islink(p):
                    continue
                st = os.lstat(p)
                nf += 1; bs += st.st_size
                if st.st_mtime > mt:
                    mt = st.st_mtime
            except OSError:
                pass
    return bs, nf, int(mt)
out = {}
for entry in sorted(os.listdir('data')):
    p = os.path.join('data', entry)
    if not os.path.isdir(p) or os.path.islink(p):
        continue
    bs, nf, mt = stats(p)
    out[f'data/{entry}'] = {'bytes': bs, 'files': nf, 'mtime': mt}
# /data00 token caches
d00 = {}
if os.path.isdir('/data00'):
    for entry in sorted(os.listdir('/data00')):
        p = os.path.join('/data00', entry)
        try:
            if os.path.isdir(p):
                bs, nf, mt = stats(p)
            else:
                bs, nf, mt = os.lstat(p).st_size, 1, int(os.lstat(p).st_mtime)
            d00[f'/data00/{entry}'] = {'bytes': bs, 'files': nf, 'mtime': mt}
        except OSError:
            pass
json.dump({'data': out, 'data00': d00}, open('/tmp/data_inv.json', 'w'))
print('data subdirs', len(out), 'data00 entries', len(d00))