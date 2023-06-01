import stanza
import time

FILENAME = "data/french_gsd.txt"

nlp = stanza.Pipeline("fr")
start_time = time.time()

f = open(FILENAME, "r")
lines = f.readlines()
f.close()

cur = 0
max_val = len(lines)

sentenceCount, tokenCount, errorCount = 0, 0, 0
confusion_matrices = {}

def handleToken(prediction, actual):
    global errorCount
    # If prediction matches actual, classify as true positive
    if prediction == actual:
        updateConfusionMatrix(actual)
    else:
        # Predicted POS is false positive
        updateConfusionMatrix(prediction, 1)
        # Actual POS is false negative
        updateConfusionMatrix(actual, 2)
        # Update statistics
        errorCount += 1

def updateConfusionMatrix(pos, ind=0):
    global tokenCount
    # Confusion matrix stored as list of 3 indices
    #   0 = True Positives (default)
    #   1 = False Positives
    #   2 = False Negatives
    #   3 = Total Count (not part of matrix)
    if pos not in confusion_matrices:
        confusion_matrices[pos] = [0]*4
    confusion_matrices[pos][ind] += 1
    # Update total counts
    confusion_matrices[pos][3] += 1
    tokenCount += 1

# Calculates F1 score given confusion matrix.
# Returns F1 score and frequency count for incrementation
def calculateF1(pos):
    matrix = confusion_matrices[pos]
    # Handle divide-by-zero errors
    if matrix[0] == 0:
        print(f"{pos}: f1 = 0 (p = 0, r = 0)")
        return 0, 0, f"{pos}: f1 = 0 | (p = 0, r = 0) | TP:{matrix[0]} FP:{matrix[1]} FN:{matrix[2]}"

    precision = matrix[0] / (matrix[0] + matrix[1])
    recall = matrix[0] / (matrix[0] + matrix[2])
    f1 = (2 * precision * recall) / (precision + recall)
    text = f"{pos} \t|   {matrix[3]} \t|   {round(f1, 5)} \t|   (p = {round(precision, 5)}, r = {round(recall, 5)}) \t| TP:{matrix[0]} FP:{matrix[1]} FN:{matrix[2]}"
    return f1, matrix[3], text

def progress_bar(cur):
    bar_width = 40
    filled_width = int(round(bar_width * cur / float(max_val)))

    # Calculate estimated time remaining (ETA)
    eta = 0
    if cur > 0:
        elapsed_time = time.time() - start_time
        eta = elapsed_time * (max_val - cur) / cur

    # Print the progress bar and ETA
    print(f'Progress: [{filled_width * "="}>{(bar_width - filled_width) * " "}] {cur}/{max_val}   ETA: {eta:.2f}', end='\r')


print("=========================================================")
print("Beginning Execution!")
print("=========================================================")

while cur < max_val:
    if not lines[cur].startswith("# text = "):
        cur += 1
        continue
    sentence = lines[cur].strip()[9:]
    while lines[cur][0] == "#":
        cur += 1
            
    # Use stanza to find token/POS pairs
    parse = nlp(sentence)
    for sentence in parse.sentences:
        for word in sentence.words:
            # Verify token matches data
            line = lines[cur].split("\t")
            if len(line) < 4:
                print(f"Weird line on {cur}: {line}")
                cur += 1
                continue
            # Skip contractions
            if "-" in line[0] or "." in line[0]:
                cur += 1
                line = lines[cur].split("\t")
                if len(line) < 4:
                    print(f"Weird line on {cur}: {line}")
                    cur += 1
                    continue
            if line[1] != word.text:
                print(f"[{cur}] - Mismatch token: should be {line[1]} but package outputted {word.text}")
                cur += 1
                continue
            
            handleToken(word.pos, line[3])
                
            cur += 1
        sentenceCount += 1
    
    # Update progress bar
    progress_bar(cur)


# Calculate F1 Scores
elapsed_time = int(time.time() - start_time)
minutes, seconds = divmod(elapsed_time, 60)

# Open the file for writing
with open('output.txt', 'a') as f:
    message = "\n\n=========================================================\n"
    message += f"Execution Completed for {FILENAME}: {sentenceCount} sentences parsed in {minutes}m {seconds}s!\n"
    message += f"\t{errorCount} errors out of {tokenCount} total tokens.\n"
    message += f"\tError rate: {round(errorCount * 1000/tokenCount)/10}%\n"
    message += "=========================================================\n"
    message += "POS Tag\t|  Freq \t|   F1 Score \t|  Precision and Recall\n"
    message += "--------------------------------------------------------\n"
    f.write(message)
    print(message, end='')
    
    completeF1 = 0
    temp = []
    for pos in confusion_matrices:
        score, freq, text = calculateF1(pos)
        completeF1 += score * (freq / tokenCount)
        temp.append((freq, text))
    temp = sorted(temp, key=lambda x: -x[0])
    
    # Print error categories sorted in descending order
    for val in temp:
        f.write(val[1] + '\n')
        print(val[1])
    
    message = "\n================================\n"
    message += f"|| Complete F1 score: {round(completeF1, 5)} ||\n"
    message += "================================\n"
    f.write(message)
    print(message, end='')
