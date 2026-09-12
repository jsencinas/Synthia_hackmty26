import json
import os
import csv

human_data = []
synthetic_data = []

# Read manifest.csv
labels = {}

with open("manifest.csv", "r") as file:
    reader = csv.DictReader(file)

    for row in reader:
        labels[row["anon_id"]] = row["label"]


# Go through every JSON file
for filename in os.listdir("turns"):

    if filename.endswith(".json"):

        # Get the anon_id from the filename
        anon_id = filename.replace(".json", "")

        # Open JSON
        with open("turns/" + filename, "r") as file:
            data = json.load(file)

        caller_turns = []
        agent_turns = []

        # Separate caller and agent
        for turn in data["turns"]:

            if turn["channel"] == 0:
                caller_turns.append(turn)
            else:
                agent_turns.append(turn)


        # Number of caller turns
        num_caller_turns = len(caller_turns)


        # Total caller speech
        total_speech = 0

        for turn in caller_turns:
            duration = turn["end"] - turn["start"]
            total_speech += duration


        # Average caller turn
        average_turn = total_speech / num_caller_turns


        # Calculate latencies
        latencies = []

        for caller in caller_turns:

            previous_agent_end = None

            for agent in agent_turns:

                if agent["end"] <= caller["start"]:

                    if previous_agent_end is None or agent["end"] > previous_agent_end:
                        previous_agent_end = agent["end"]

            if previous_agent_end is not None:
                latency = caller["start"] - previous_agent_end
                latencies.append(latency)


        # Latency statistics
        minimum_latency = min(latencies)
        average_latency = sum(latencies) / len(latencies)
        maximum_latency = max(latencies)


        # Create feature vector
        features = [
            num_caller_turns,
            total_speech,
            average_turn,
            minimum_latency,
            average_latency,
            maximum_latency
        ]


        # Get the correct label from manifest
        label = labels[anon_id]


        # Put the data into the correct group
        if label == "human":
            human_data.append(features)

        else:
            synthetic_data.append(features)


# Show results
print("Number of human calls:", len(human_data))
print("Number of synthetic calls:", len(synthetic_data))

print("\nFirst human example:")
print(human_data[0])

print("\nFirst synthetic example:")
print(synthetic_data[0])

# Calculate averages for human calls

human_avg = []

for i in range(6):
    total = 0

    for data in human_data:
        total += data[i]

    average = total / len(human_data)
    human_avg.append(average)


# Calculate averages for synthetic calls

synthetic_avg = []

for i in range(6):
    total = 0

    for data in synthetic_data:
        total += data[i]

    average = total / len(synthetic_data)
    synthetic_avg.append(average)


# Feature names

feature_names = [
    "Caller turns",
    "Total speech",
    "Average turn",
    "Minimum latency",
    "Average latency",
    "Maximum latency"
]


# Print final results

print("\n========== FINAL RESULTS ==========")

print("\nHUMAN AVERAGES:")

for i in range(6):
    print(feature_names[i], ":", round(human_avg[i], 3))


print("\nSYNTHETIC AVERAGES:")

for i in range(6):
    print(feature_names[i], ":", round(synthetic_avg[i], 3))