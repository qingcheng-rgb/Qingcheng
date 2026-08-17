import glob, json, os, re, pypdf
S = '/var/www/python/Qingcheng/QCTest/Manual_bidding/logs/scratch'
pdfs = sorted(glob.glob('/var/www/python/Qingcheng/QCTest/Manual_bidding/logs/*.pdf'))
assert pdfs, 'no attribution PDF in logs/'
pdf = pdfs[-1]
print('using:', os.path.basename(pdf))
txt = '\n'.join((p.extract_text() or '') for p in pypdf.PdfReader(pdf).pages)
open(S + '/attribution.txt', 'w').write(txt)

hdrs = []
for m in re.finditer(r'Monitored Element:\s*', txt):
    nm = re.match(r'[a-z0-9_.\-]+', re.sub(r'\s+', '', txt[m.end():m.end() + 80]))
    if nm:
        hdrs.append((m.start(), nm.group(0)))
sec = {}
for i, (pos, nm) in enumerate(hdrs):
    sec.setdefault(nm, txt[pos:hdrs[i + 1][0] if i + 1 < len(hdrs) else len(txt)])
print('sections in PDF: {}'.format(len(hdrs)))
print('section names: {}'.format(sorted(sec)))

names = json.load(open(S + '/names.json'))
for name in names:
    body = sec.get(name)
    if body is None:
        print('\n{}: NOT IN PDF'.format(name)); continue
    d = re.search(r'Analysis Date:\s*([\d\-]+)', body)
    print('\n' + '=' * 70 + '\n{}  (analysis date {})'.format(name, d.group(1) if d else '?'))
    for lbl, pat in (('SUMMARY',  r'3\.1 Summary:.*?(?=3\.2|\Z)'),
                     ('OUTAGES',  r'3\.3 Outages Table:(.*?)(?=3\.4|\Z)'),
                     ('WEATHER',  r'3\.4 Weather Drivers:(.*?)(?=3\.5|\Z)')):
        m = re.search(pat, body, re.S)
        if m:
            print(' {}: {}'.format(lbl, re.sub(r'\s+', ' ', m.group(0))[:1400]))
