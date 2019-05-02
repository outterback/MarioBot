import npyscreen
from datetime import datetime
from uuid import uuid4

class DateForm(npyscreen.Form):
    def while_waiting(self):
        #npyscreen.notify_wait("Update")
        #self.date_widget.value = datetime.now()
        self.values.append(str(uuid4())[0:16] + '\n')

        strs = ''
        for v in self.values[-10:-1]:
            strs += v
        self.date_widget.value = strs # self.values[-5:-1]
        self.display()


    def create(self):
        self.date_widget = self.add(npyscreen.MultiLineEdit,
                                    value='', editable=False)
        self.values = []



class DateApp(npyscreen.NPSAppManaged):

    keypress_timeout_default = 1

    def onStart(self):
        self.addForm("MAIN", DateForm, name="Time")

if __name__ == '__main__':
    app = DateApp()
    app.run()