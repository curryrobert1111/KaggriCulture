"""Create a variant agent file with PARAMS overrides.
usage: python variant.py out.py '{"cap_lambda":0.3}' [base=main.py]"""
import json, sys
out, ov = sys.argv[1], json.loads(sys.argv[2])
base = sys.argv[3] if len(sys.argv) > 3 else "main.py"
src = open(base).read()
src += "\n\nPARAMS.update(%r)\n" % ov
open(out, "w").write(src)
