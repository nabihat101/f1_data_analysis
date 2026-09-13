import fastf1

YEAR = 2025
TRACK = "Monaco"

fp1 = fastf1.get_session(YEAR, TRACK, "FP1")
fp2 = fastf1.get_session(YEAR, TRACK, "FP2")

fp1.load()
fp2.load()

print("\nFP1 results:")
print(fp1.results[["Abbreviation", "Position"]].head(10))

print("\nFP2 results:")
print(fp2.results[["Abbreviation", "Position"]].head(10))