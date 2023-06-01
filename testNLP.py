import stanza

stanza.download("en")
nlp = stanza.Pipeline("en")

text = "“While much of the digital transition is unprecedented in the United States, the peaceful transition of power is not,” Obama special assistant Kori Schulman wrote in a blog post Monday."
doc = nlp(text)
print("=========================")
print(doc)
print("=========================")
for sentence in doc.sentences:
    for word in sentence.words:
        print(word.text, word.pos)
print("=========================")
print(doc.entities)

# print("=========================")
# print("=========================")
# text = "Jacob Collier is a Grammy-awarded English artist from London."
# doc = nlp(text)
# print("=========================")
# print(doc)
# print("=========================")
# for sentence in doc.sentences:
#     for word in sentence.words:
#         print(word.text, word.pos)
# print("=========================")
# print(doc.entities)
