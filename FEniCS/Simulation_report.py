"""
Created on Mon Aug 19 13:00:00 2019
@author: SantiagoOrtiz
"""
import os

def Execution_time(Start_time, End_time, folder,
                   VisVar='', Print=False):

    if Print:
        print('\n'+"<-------The 2D simulation execution:------->")
        print("Got underway at ", Start_time)
        print("Concluded at ", End_time)
        print("Took around", End_time-Start_time, '\n')

    def days_hours_minutes(td):
            return str(td.days) + " days - " + \
                   str(td.seconds//3600) + " hours - " + \
                   str((td.seconds//60) % 60) + " minutes."

    # print(days_hours_minutes(End_time-Start_time))

    SED_times = [Start_time.strftime("%d-%b (%H:%M:%S)"),
                 End_time.strftime("%d-%b (%H:%M:%S)"),
                 days_hours_minutes(End_time-Start_time)]
    messages = ["Got underway at %s\n", "Conluded at %s\n",
                "Took around %s\n"]

    nameTXT = VisVar.replace("/", "")
    f = open(os.path.join(folder,nameTXT+".txt"), "w+")

    f.write('\n'+"<-------The  2D simulation execution:------->"+'\n')
    for msg, SED in zip(messages, SED_times):
        f.write(msg % (SED))
