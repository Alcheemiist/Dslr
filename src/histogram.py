
import pandas as pd
import matplotlib.pyplot as plt


def separate_houses(data):
    houses_columns = ["Ravenclaw", "Gryffindor", "Slytherin", "Hufflepuff"]

    Ravenclaw = data[data["Hogwarts House"] == "Ravenclaw"]
    print("Ravenclaw house : ", len(Ravenclaw))

    Gryffindor = data[data["Hogwarts House"] == "Gryffindor"]
    print("Gryffindor house : ",len( Gryffindor))

    Slytherin = data[data["Hogwarts House"] == "Slytherin"]
    print("Slytherin house : ", len(Slytherin))

    Hufflepuff = data[data["Hogwarts House"] == "Hufflepuff"]
    print("Hufflepuff house : ", len(Hufflepuff))

    return Ravenclaw, Gryffindor, Slytherin, Hufflepuff

if __name__ == "__main__":
    print("This is histogram.py")
    # Read Data
    input_file = "../datasets/dataset_train.csv"
    data = pd.read_csv(input_file)

    print("len data : ", len(data), "\n")

    # Separate Data into houses
    Ravenclaw, Gryffindor, Slytherin, Hufflepuff = separate_houses(data)

    
    # Get metrics for each house and sort them by standard deviation
    Ravenclaw_metrics = Ravenclaw.describe().loc[["mean", "std"]].T
    Gryffindor_metrics = Gryffindor.describe().loc[["mean", "std"]].T
    Slytherin_metrics = Slytherin.describe().loc[["mean", "std"]].T
    Hufflepuff_metrics = Hufflepuff.describe().loc[["mean", "std"]].T

    Ravenclaw_metrics.sort_values(by="std", ascending=True, inplace=True)
    Gryffindor_metrics.sort_values(by="std", ascending=True, inplace=True)
    Slytherin_metrics.sort_values(by="std", ascending=True, inplace=True)
    Hufflepuff_metrics.sort_values(by="std", ascending=True, inplace=True)

    # print("Ravenclaw_metrics : \n", Ravenclaw_metrics.iloc[0])
    # print("\nGryffindor_metrics : \n", Gryffindor_metrics.iloc[0])
    # print("\nSlytherin_metrics : \n", Slytherin_metrics.iloc[0])
    # print("\nHufflepuff_metrics : \n", Hufflepuff_metrics.iloc[0])

    courses = ['Arithmancy', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts',
       'Divination', 'Muggle Studies', 'Ancient Runes', 'History of Magic',
       'Transfiguration', 'Potions', 'Care of Magical Creatures', 'Charms',
       'Flying']
    Ravenclaw.drop(columns=['Index', 'Hogwarts House', 'First Name', 'Last Name', 'Birthday','Best Hand',], inplace=True)
    print(Ravenclaw.columns)

    plt.figure(figsize=(10, 8))
    plt.barh(courses, Ravenclaw_metrics['std'])
    plt.xlabel('Standard Deviation')
    plt.show()
    exit()
    plt.title('Standard Deviation of Course Scores for Ravenclaw')
    plt.gca().invert_yaxis()  # Invert y axis for better readability
    plt.show()


    plt.show()
    
