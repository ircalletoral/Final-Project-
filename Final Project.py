#Import current data usage inclusign logistic regression
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

#Create Directory and Idetify Folder with Data
current_directory = os.getcwd
#print(current_directory)

new_directory_path = r"C:\Users\ircal\Desktop\Coding\Health Python"
os.chdir(new_directory_path)

hospital_dir = os.getcwd()
#print(hospital_dir)

#Obtain Hospital Data from 1st Hospital
hop_1 = 'Hospital1.txt'
df_1 = pd.read_csv(hop_1)

#Testing for Columns and Data Visualization, will be a comment on final code
#print(df_1.columns)

#Hospital 1 Data Analysis and statistics
hop_1_readd = np.sum(df_1[' Readmission'])
hop_1_StaffSat = np.mean(df_1[' StaffSatisfaction'])
hop_1_CleanSat = np.mean(df_1[" CleanlinessSatisfaction"])
hop_1_FoodSat = np.mean(df_1[' FoodSatisfaction'])
hop_1_ComfSat = np.mean(df_1[' ComfortSatisfaction'])
hop_1_CommSat = np.mean(df_1[' CommunicationSatisfaction'])

#Calculate Overall Satisfaction for Hospital 1
df_1 = df_1.convert_dtypes()
df_1[" OverallSatisfaction"] = df_1[[' StaffSatisfaction', " CleanlinessSatisfaction", ' FoodSatisfaction', ' ComfortSatisfaction', ' CommunicationSatisfaction']].mean(axis = 1)
plt.boxplot(df_1[" OverallSatisfaction"], showfliers = True)

hop_1_OvSat = np.mean(df_1[" OverallSatisfaction"])

#Transforming the data to binary and creating the Log Regression
x = df_1[" OverallSatisfaction"].values.reshape(-1, 1)
y = df_1[" Readmission"]
log_reg_h1 = LogisticRegression().fit(x, y)

###########################################################
# We will make calculators to make this cleaner and easier.
###########################################################

# Correlation Coefficient caluclator - IT WORKS
# For use just type correlation_calc(logistic regression name)
def correlation_calc(h_n):
    correlation_coefficient = h_n.coef_[0][0]
    correlation_text = " "

    if correlation_coefficient > 0:
        print("Logistic regression results indicate a: ")
        if correlation_coefficient > 0.7:
            correlation_text = "Strong correlation"
        elif correlation_coefficient > 0.5:
            correlation_text = "Moderate correlation"
        else:
            correlation_text = "Weak Correlation"
    else:
        print("Logistic Regression Results Indicated:")
        correlation_text = "No correlation"

    print(correlation_text)

    return correlation_coefficient

    print(f"Correlation Coefficient was: {correlation_coefficient}")

#Plotting the data for hospitals - IT WORKS!!
def data_plot_hn(hospital_number):
    plt.scatter(x, y)
    plt.xlabel("Overall Satisfaction Scores")
    plt.ylabel("Logistic Regression - Overall Satisfaction vs Readmission")
    plt.plot(x, hospital_number.predict(x), color = 'green')
    plt.xlim(2, 5)

#Get the data for hospital 1 correlation
correlation_hop1 = correlation_calc(log_reg_h1)

#Obtain Data from the Second Hospital
hop_2 = "Hospital2.txt"
df_2 = pd.read_csv(hop_2)
#print(df_2.columns)

#Hospital 2 data statistics and numbers
hop_2_readd = np.sum(df_2[' Readmission'])
hop_2_StaffSat = np.mean(df_2[' StaffSatisfaction'])
hop_2_CleanSat = np.mean(df_2[" CleanlinessSatisfaction"])
hop_2_FoodSat = np.mean(df_2[' FoodSatisfaction'])
hop_2_ComfSat = np.mean(df_2[' ComfortSatisfaction'])
hop_2_CommSat = np.mean(df_2[' CommunicationSatisfaction'])

#Calculate Overall Satisfaction for Hospital 2
df_2 = df_2.convert_dtypes()
df_2[" OverallSatisfaction"] = df_2[[' StaffSatisfaction', " CleanlinessSatisfaction", ' FoodSatisfaction', ' ComfortSatisfaction', ' CommunicationSatisfaction']].mean(axis = 1)
plt.boxplot(df_2[" OverallSatisfaction"], showfliers = True)

#Overall Average Satisfaction for Hospital 2
hop_2_OvSat = np.mean(df_2[" OverallSatisfaction"])

#Log reg calc will come later for hospital 2

#No need to make more calculations bc we made the calculator already!!
print("Hospital Comparisons:")
print("----------------------")
print(" ")

#Printing all Hospital 1 Data

print("Hospital 1 Data Analysis:")
print("-------------------------")

print(f"Number of readmissions for Hospital 1 are: {hop_1_readd}")
hop_1_StaffSat = round(hop_1_StaffSat, 2)
print(f"The average Staff Satisfaction is:{hop_1_StaffSat}")
hop_1_CleanSat = round(hop_1_CleanSat, 2)
print(f"The average Cleanliness Satisfaction is:{hop_1_CleanSat}")
hop_1_FoodSat = round(hop_1_FoodSat, 2)
print(f"The average Food Satisfaction is:{hop_1_FoodSat}")
hop_1_ComfSat = round(hop_1_ComfSat, 2)
print(f"The average Comfort Satisfaction is:{hop_1_ComfSat}")
hop_1_CommSat = round(hop_1_CommSat, 2)
print(f"The average Communications Satisfaction is:{hop_1_CommSat}")

print("Logistic Regression Result:")

data_plot_hn(log_reg_h1)

correlation_calc(log_reg_h1)

print("See below for a graphed model of this prediction")

#Transforming the data to binary and creating Log Regression for Hospital 2
x = df_2[" OverallSatisfaction"].values.reshape(-1, 1)
y = df_2[" Readmission"]
log_reg_h2 = LogisticRegression().fit(x, y)

#Printing all Hospital 2 Data



print("Hospital 2 Data Analysis:")
print("-------------------------")

print(f"Number of readmissions for Hospital 2 are: {hop_2_readd}")
hop_2_StaffSat = round(hop_2_StaffSat, 2)
print(f"The average Staff Satisfaction is:{hop_2_StaffSat}")
hop_2_CleanSat = round(hop_2_CleanSat, 2)
print(f"The average Cleanliness Satisfaction is:{hop_2_CleanSat}")
hop_2_FoodSat = round(hop_2_FoodSat, 2)
print(f"The average Food Satisfaction is:{hop_2_FoodSat}")
hop_2_ComfSat = round(hop_2_ComfSat, 2)
print(f"The average Comfort Satisfaction is:{hop_2_ComfSat}")
hop_2_CommSat = round(hop_2_CommSat, 2)
print(f"The average Communications Satisfaction is:{hop_2_CommSat}")

print("Logistic Regression Result:")

data_plot_hn(log_reg_h2)

#Getting the coorelation value for Hop 2
correlation_hop2 = correlation_calc(log_reg_h2)

print("See below for a graphed model of this prediction")

#Hospital Comparison calculator
# overall_rating_comparison(average satisfaction 1, average satisfaction 2, log regresion 1, log regresion 2)

def overall_rating_comparison(hopn, hopn2, corr_hp1, corr_hp2):
    sat_rating = " "
    # Make overall satisfaction comparison
    if (hopn - hopn2) >= 1:
        sat_rating = "considerably higher satisfaction"
        conclusion = "considerably better"
    elif (hopn - hopn2) > 0 and (hopn - hopn2) < 1:
        sat_rating = "slightly higher satisfaction"
        conclusion = "slightly better"
    elif (hopn - hopn2) < (-1):
        sat_rating = "considerably lower satisfaction"
        conclusion = "considerably worse"
    elif (hopn - hopn2) == 0:
        sat_rating = "exact same satisfaction"
        conclusion = "the exact same"
    else:
        sat_rating = "slightly lower satisfaction"
        conclusion = "slightly worse"

    # Making rating comparison

    correlation_comp = corr_hp1 - corr_hp2

    if correlation_comp > 0 or correlation_comp < 0:
        if correlation_comp > 0.5:
            correlation_comp_text = "Strong"
        elif correlation_comp > 0.2:
            correlation_comp_text = "Moderate"
        elif correlation_comp - 0.2:
            correlation_comp_text = "Weak"
        elif correlation_comp > -0.5:
            correlation_comp_text = "Very Weak"
    else:
        correlation_comp_text = "no"

    print(
        f"Based on the data analysis and logistic regression results, Hospital 1 has a {sat_rating} rating and a {correlation_comp_text} correlation between overall satisfaction scores and readmission rates compared to Hospital 2.")
    print(
        f"However, both have relatively low readmission rates, indicating that they are performing well in patient care and satisfaction")
    print(" ")
    print(
        f"Conclusion: Hospital 1 may be doing {conclusion} in terms of patient satisfaction and readmission rates compared to Hosptial 2")


print("Hospital Comparisons: ")
print("----------------------")
print(" ")

overall_rating_comparison(hop_1_OvSat, hop_2_OvSat, correlation_hop1, correlation_hop2)