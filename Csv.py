import csv

class Csv:

    def __init__(self, dataToWrite):
        self.dataToWrite = dataToWrite

    # write lines into a file, given his name
    def writeToCsv(self, fileName):
        # create the csv writer
        f = open(fileName, 'w', newline='')
        writer = csv.writer(f)
        writer.writerow(["genre"])
        for genre in self.dataToWrite:
            writer.writerow([genre])
        f.close()

    # clear a csv file given his name
    def clearCsv(self, fileName):
        f = open(fileName, 'w')
        f.truncate(0)