#  -------------------------------------------------------------
#  -------------------------------------------------------------
#  -------------------------------------------------------------
#  ----------------------- SALARY FINDER -----------------------
#  -------------------------------------------------------------
#  -------------------------------------------------------------
#  -------------------------------------------------------------


# IMPORTING REQUIRED LIBRARIES

import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

import warnings
warnings.filterwarnings('ignore')


# CREATING OUR DATASET USING PANDAS

rawdata = {
'CGPA': [9.1, 8.5, 7.0, 8.2, 9.5, 6.8, 7.8, 8.0, 7.2, 9.0],
'Internship': [1, 1, 0, 1, 1, 0, 1, 0, 0, 1],       # 0 = False, 1 = True
'College_Tier': [1, 1, 2, 2, 1, 3, 2, 3, 2, 1],     # 1 = Top tier college, 2 = Medium tier college, 3 = Low tier college``
'Salary (LPA)': [22, 18, 8, 16, 24, 6, 10, 7, 9, 21]
}

data = pd.DataFrame(rawdata)


# TRAINING & TESTING MODEL FROM OUR DATASETS USING SCIKIT-LEARN

x = data[['CGPA', 'Internship', 'College_Tier']]
y = data['Salary (LPA)']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)


# USING MODEL FOR USER'S SALARY PREDICTION

def predictSalary(cgpa, internship, collegeTier) :
    return model.predict([[cgpa, internship, collegeTier]])[0]

while True:
    print('\n========= Welcome to Salary Finder =========\n')
    
    # Main Menu
    print('1. Add your details and predict salary')
    print('2. View sample format')
    print('3. About this tool')
    print('4. Exit')

    userChoice = input('Enter your choice: ')

    match userChoice:
        

        # User Input for Salary Prediction
        case '1':
            try:
                print("\n🔍 Please provide the following details:")

                cgpa = float(input('🌟 Enter your CGPA: '))
                internship = int(input('💼 How many internships you have done: '))
                collegeTier = int(input('🏫 Your College tier (1, 2, or 3): '))

                salary = predictSalary(cgpa, internship, collegeTier)
                print(f'\n🎯 Based on the details you provided, your estimated salary package is: {int(salary)} LPA')
                print('\n**Please note that this is an estimation based on your CGPA, internships, and college tier. '
                    'The actual salary may vary depending on various factors like company, role, and market conditions.')

            except Exception as e:
                print(f'⚠️ Error: {e}')

            input("\nPress Enter to return to the menu...")


        # Display Sample Format
        case '2':
            print("\n📄 Sample Format:")
            print("CGPA: 8.5")
            print("Internships: 2")
            print("College Tier: 1")

            print("\n🏫 College Tier Guide:")
            print("1️⃣  Tier 1 - Top colleges (IITs, NITs, IIITs, BITS, etc.)")
            print("2️⃣  Tier 2 - State universities, renowned private colleges")
            print("3️⃣  Tier 3 - Local colleges or institutes with average placements")

            input("\nPress Enter to return to the menu...")


        # About the tool
        case '3':
            print("\nℹ️  About Salary Finder Tool")
            print("This tool helps you estimate your expected salary based on the following factors:")
            print("✔️  Your CGPA")
            print("✔️  Number of internships you've completed")
            print("✔️  The tier of your college")

            print("\n🤖 Powered by a trained Machine Learning model (like Linear Regression),")
            print("it gives you a rough idea of your potential salary package in LPA.")
            print("Perfect for freshers preparing for placements!")

            print("\n✨ Made by Jinit Limbachiya ✨")

            input("\nPress Enter to return to the menu...")

        # Exit the program
        case '4':
            print('\n👋 Exiting... Have a nice day!')
            exit()

        case _:
            print('❌ Invalid Option. Please choose a number between 1 and 4')

            input("\nPress Enter to return to the menu...")