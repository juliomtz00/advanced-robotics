import sys

def findLargestNumber(listOfNumbers):
    if not listOfNumbers:
        return None

    largest = listOfNumbers[0]
    for i in listOfNumbers:
        if i > largest:
            largest = i

    return largest

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python largestNumber.py <comma-separated-list-of-numbers>")
        sys.exit(1)

    numbersString = sys.argv[1]
    listOfNumbers = [int(num) for num in numbersString.split(',')]
    largestInteger = findLargestNumber(listOfNumbers)
    print("The largest number is:", largestInteger)