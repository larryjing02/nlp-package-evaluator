import stanza

FILENAME = "data/french_gsd.txt"

# stanza.download("en")
nlp = stanza.Pipeline("fr")

f = open(FILENAME, "r")
lines = f.readlines()
f.close()

cur = 0

sentenceCount, tokenCount, errorCount = 0, 0, 0

while cur < len(lines):
    if not lines[cur].startswith("# text = "):
        cur += 1
        continue
    sentence = lines[cur].strip()[9:]
    cur += 1
    while lines[cur][0] == "#":
        cur += 1
        
    #  Use stanza to find token/POS pairs
    parse = nlp(sentence)
    
    for sentence in parse.sentences:
        for word in sentence.words:
            # Verify token matches data
            line = lines[cur].split("\t")
            if len(line) < 4:
                print(f"Weird line on {cur}: {line}")
                cur += 1
                continue
            # Token mismatch (represents tokenization errors)
            if line[1] != word.text:
                print(f"[{cur}] - Mismatch token: should be {line[1]} but package outputted {word.text}")
                cur += 1
                continue
            # POS mismatch (represents data parsing errors)
            if line[3] != word.pos:
                print(f"[{cur}] - Mismatch POS: {line[1]} should be {line[3]} but package outputted {word.pos}")
                errorCount += 1
            cur += 1
            tokenCount += 1
        sentenceCount += 1

print("=========================================================")
print(f"Execution Completed: {sentenceCount} sentences parsed!")
print(f"\t{errorCount} errors out of {tokenCount} total tokens.")
print(f"\tError rate: {round(errorCount * 1000/tokenCount)/10}%")
