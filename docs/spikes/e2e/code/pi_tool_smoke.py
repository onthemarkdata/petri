import sys
import time
sys.path.insert(0, "/tmp/claude-0/-home-user-petri/767427e9-1a07-5e63-aad4-1e3d3de99d01/scratchpad")
from pi_provider_scratch import pi_ask, PiAskError

PROMPT = (
    "Use the WebSearch tool to find sources on whether regular expressions can "
    "parse HTML, then use WebFetch to read the top source. Then answer in 2 "
    "sentences and cite the exact URL you fetched."
)
t = time.time()
try:
    r = pi_ask(PROMPT, provider="anthropic", model="anthropic/claude-sonnet-4-6",
               pi_bin="/opt/node22/bin/pi", timeout=160, tools=["WebSearch", "WebFetch"])
    print("SUCCESS:")
    print(r.text[:1000])
    hits = [d for d in ("wikipedia.org", "johndcook", "cmu.edu", "odu.edu", "neilmadden", "codinghorror") if d in r.text]
    print("--- cited real corpus URL(s):", hits)
except PiAskError as e:
    print("FAIL channel=%s msg=%r" % (e.channel, e.error_message[:400]))
print("elapsed %.1fs" % (time.time() - t))
