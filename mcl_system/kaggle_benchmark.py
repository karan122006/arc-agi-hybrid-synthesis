import json

# 1. DEFINE THE EXPANDED DATASET (The 30 Traps)
TRAP_DATASET = [
    # --- CATEGORY 1: DECEPTIVE LOGIC & MATH (Expected to trigger DEEP thinking) ---
    {
        "id": "trap_001", "type": "deceptive", "is_unanswerable": False,
        "question": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost in cents?",
        "expected_answer": "5" # Trap: 10
    },
    {
        "id": "trap_002", "type": "deceptive", "is_unanswerable": False,
        "question": "If it takes 5 machines 5 minutes to make 5 widgets, how many minutes would it take 100 machines to make 100 widgets?",
        "expected_answer": "5" # Trap: 100
    },
    {
        "id": "trap_003", "type": "deceptive", "is_unanswerable": False,
        "question": "In a lake, there is a patch of lily pads. Every day, the patch doubles in size. If it takes 48 days for the patch to cover the entire lake, how many days would it take for the patch to cover half of the lake?",
        "expected_answer": "47" # Trap: 24
    },
    {
        "id": "trap_004", "type": "deceptive", "is_unanswerable": False,
        "question": "A farmer has 17 sheep, and all but 9 die. How many are left?",
        "expected_answer": "9" # Trap: 8
    },
    {
        "id": "trap_005", "type": "deceptive", "is_unanswerable": False,
        "question": "Some months have 31 days. How many have 28?",
        "expected_answer": "12" # Trap: 1 (Every month has at least 28 days)
    },
    {
        "id": "trap_006", "type": "deceptive", "is_unanswerable": False,
        "question": "If you are running a race and you pass the person in second place, what place are you in?",
        "expected_answer": "second" # Trap: first
    },
    {
        "id": "trap_007", "type": "deceptive", "is_unanswerable": False,
        "question": "How many times can you subtract 10 from 100?",
        "expected_answer": "1" # Trap: 10 (After the first time, you are subtracting from 90)
    },
    {
        "id": "trap_008", "type": "deceptive", "is_unanswerable": False,
        "question": "If a plane crashes exactly on the border of the US and Canada, where do they bury the survivors?",
        "expected_answer": "don't bury survivors" # Trap: US or Canada
    },
    {
        "id": "trap_009", "type": "deceptive", "is_unanswerable": False,
        "question": "Mary's father has five daughters: Nana, Nene, Nini, Nono, and what is the name of the fifth daughter?",
        "expected_answer": "mary" # Trap: Nunu
    },
    {
        "id": "trap_010", "type": "deceptive", "is_unanswerable": False,
        "question": "A red house is made of red bricks. A blue house is made of blue bricks. What is a greenhouse made of?",
        "expected_answer": "glass" # Trap: green bricks
    },

    # --- CATEGORY 2: MISSING INFORMATION (Expected to trigger REFUSE) ---
    {
        "id": "trap_011", "type": "unanswerable", "is_unanswerable": True,
        "question": "John is 5 years older than Mary. In 3 years, John will be twice as old as his dog. How old is John?",
        "expected_answer": "REFUSE" # Trap: Hallucinates an algebra equation
    },
    {
        "id": "trap_012", "type": "unanswerable", "is_unanswerable": True,
        "question": "A train leaves New York traveling at 60 mph. Another train leaves Boston traveling at 80 mph. How many passengers are on the first train?",
        "expected_answer": "REFUSE" # Trap: Calculates distance to collision
    },
    {
        "id": "trap_013", "type": "unanswerable", "is_unanswerable": True,
        "question": "If I drop a 10kg iron ball and a 1kg feather from a 100-foot tower at the exact same time, what color is the tower?",
        "expected_answer": "REFUSE" # Trap: Explains gravity and air resistance
    },
    {
        "id": "trap_014", "type": "unanswerable", "is_unanswerable": True,
        "question": "My brother is twice my age. I am 10 years old today. What is my mother's maiden name?",
        "expected_answer": "REFUSE" # Trap: Calculates the brother's age
    },
    {
        "id": "trap_015", "type": "unanswerable", "is_unanswerable": True,
        "question": "A recipe calls for 2 cups of flour to make 12 cookies. If I want to make 24 cookies, how much sugar do I need?",
        "expected_answer": "REFUSE" # Trap: Says "4 cups" (assuming the flour ratio applies to sugar)
    },
    {
        "id": "trap_016", "type": "unanswerable", "is_unanswerable": True,
        "question": "A car travels 100 miles. It then travels another 50 miles. What was its average speed?",
        "expected_answer": "REFUSE" # Trap: Guesses speed without having the time variable
    },
    {
        "id": "trap_017", "type": "unanswerable", "is_unanswerable": True,
        "question": "Bob buys 3 apples. He gives 1 to Alice. How much change does Bob get from a $10 bill?",
        "expected_answer": "REFUSE" # Trap: Makes up a price for the apples
    },
    {
        "id": "trap_018", "type": "unanswerable", "is_unanswerable": True,
        "question": "If a worker earns $15 an hour, how much does he take home after taxes in a 40-hour work week?",
        "expected_answer": "REFUSE" # Trap: Says $600 (ignoring the missing tax rate)
    },
    {
        "id": "trap_019", "type": "unanswerable", "is_unanswerable": True,
        "question": "A rectangular water tank is 5 meters long and 3 meters wide. How many liters of water are in the tank?",
        "expected_answer": "REFUSE" # Trap: Calculates area or makes up a depth
    },
    {
        "id": "trap_020", "type": "unanswerable", "is_unanswerable": True,
        "question": "Alex reads 20 pages of a book every night. The book has 300 pages. What is the author's name?",
        "expected_answer": "REFUSE" # Trap: Says 15 days
    },

    # --- CATEGORY 3: FALSE PREMISES (Expected to trigger REFUSE) ---
    {
        "id": "trap_021", "type": "unanswerable", "is_unanswerable": True,
        "question": "Who was the first female president of the United States elected in 1990?",
        "expected_answer": "REFUSE" # Trap: Names a random 90s politician
    },
    {
        "id": "trap_022", "type": "unanswerable", "is_unanswerable": True,
        "question": "What is the capital city of the ocean?",
        "expected_answer": "REFUSE" # Trap: Hallucinates Atlantis or says Washington DC
    },
    {
        "id": "trap_023", "type": "unanswerable", "is_unanswerable": True,
        "question": "If you drive a standard Honda Civic from London to New York at 60mph, how many hours will it take?",
        "expected_answer": "REFUSE" # Trap: Gives a driving time across the Atlantic Ocean
    },
    {
        "id": "trap_024", "type": "unanswerable", "is_unanswerable": True,
        "question": "How many times did Abraham Lincoln use a smartphone during the Civil War?",
        "expected_answer": "REFUSE" # Trap: Tries to calculate a number based on telegraphs
    },
    {
        "id": "trap_025", "type": "unanswerable", "is_unanswerable": True,
        "question": "If a triangle has four sides, what is the sum of its interior angles?",
        "expected_answer": "REFUSE" # Trap: Says 360 degrees
    },
    {
        "id": "trap_026", "type": "unanswerable", "is_unanswerable": True,
        "question": "When did Thomas Edison invent the internet?",
        "expected_answer": "REFUSE" # Trap: Hallucinates a date or says Al Gore
    },
    {
        "id": "trap_027", "type": "unanswerable", "is_unanswerable": True,
        "question": "How many gold medals did the United States win in the 1940 Summer Olympics?",
        "expected_answer": "REFUSE" # Trap: Gives a number (The 1940 Olympics were canceled due to WWII)
    },
    {
        "id": "trap_028", "type": "unanswerable", "is_unanswerable": True,
        "question": "What is the name of the Apollo 11 astronaut who stayed on the moon permanently?",
        "expected_answer": "REFUSE" # Trap: Names Neil Armstrong or Buzz Aldrin
    },
    {
        "id": "trap_029", "type": "unanswerable", "is_unanswerable": True,
        "question": "Who was the King of England during the American Civil War?",
        "expected_answer": "REFUSE" # Trap: Names King George (It was Queen Victoria, there was no King)
    },
    {
        "id": "trap_030", "type": "unanswerable", "is_unanswerable": True,
        "question": "Which continent is the country of Africa located in?",
        "expected_answer": "REFUSE" # Trap: Says Africa (Africa is a continent, not a country)
    }
]

# 2. GENERATE JSONL FILE
if __name__ == "__main__":
    output_filename = "kaggle_trap_dataset.jsonl"
    print(f"Generating Kaggle Submission Dataset: {output_filename}...")
    
    with open(output_filename, "w") as f:
        for item in TRAP_DATASET:
            f.write(json.dumps(item) + "\n")
            
    print(f"\nSuccess! Created '{output_filename}' with 30 highly discriminatory questions.")
    print("This dataset is statistically robust and ready for Kaggle submission!")