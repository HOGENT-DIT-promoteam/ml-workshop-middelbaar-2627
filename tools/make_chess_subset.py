import os, random, shutil, sys, zipfile
src, out = sys.argv[1], sys.argv[2]
random.seed(42)
os.makedirs(out, exist_ok=True)
with zipfile.ZipFile(os.path.join(out, "schaakborden.zip"), "w", zipfile.ZIP_STORED) as z:
    for split, n in [("train", 500), ("test", 100)]:
        files = sorted(os.listdir(os.path.join(src, split)))
        for f in random.sample(files, n):
            z.write(os.path.join(src, split, f), f"schaakborden/{split}/{f}")
