import os
from kivy.app import App
#from kivymd.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.core.window import Window
from kivy.clock import Clock
from kivy.uix.image import Image
from kivy.uix.boxlayout import BoxLayout
from kivy.animation import Animation
from kivy.properties import ObjectProperty
from kivy.uix.popup import Popup
from kivy.uix.behaviors import TouchRippleBehavior
from kivy.uix.button import Button
from kivy.utils import get_color_from_hex
from kivy.uix.label import Label
#from kivymd.uix.label import MDLabel
#from kivymd.uix.button import MDRectangleFlatButton
from database import DataBase
from poseEstimationWithKerasModel import pose_classification
#from estimationwithModels import PoseCalculation


current_dir = os.path.dirname(os.path.realpath(__file__))
Window.size = (400, 600)
Window.clearcolor = (64/255, 64/255, 64/255, 1)

class RectangleFlatButton(TouchRippleBehavior, Button):
    primary_color = get_color_from_hex("#EB8933")



class SplashScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        anim = Animation(background_color=(9/255, 121/255, 105/255,1))
        anim.repeat = True

        self.img = Image(source=os.path.join(current_dir, "spash_image.png"))
       # print(os.path.join(current_dir))
        box_layout = BoxLayout()
        self.add_widget(box_layout)
        box_layout.add_widget(self.img)

    def on_enter(self):
        self.img.opacity = 0
        animation = Animation(duration=3, opacity=1) + Animation(duration=3, opacity=0)
        animation.start(self.img)

class CreateAccountWindow(Screen):
    namee = ObjectProperty(None)
    email = ObjectProperty(None)
    password = ObjectProperty(None)

    def submit(self):
        if self.namee.text != "" and self.email.text != "" and self.email.text.count("@") == 1 and self.email.text.count(".") > 0:
            if self.password != "":
                db.add_user(self.email.text, self.password.text, self.namee.text)

                self.reset()

                sm.current = "login"
            else:
                invalidForm()
        else:
            invalidForm()

    def login(self):
        self.reset()
        sm.current = "login"

    def reset(self):
        self.email.text = ""
        self.password.text = ""
        self.namee.text = ""


class LoginWindow(Screen):
    email = ObjectProperty(None)
    password = ObjectProperty(None)

    def loginBtn(self):
        if db.validate(self.email.text, self.password.text):
            MainWindow.current = self.email.text
            self.reset()
            sm.current = "mainPage"
        else:
            invalidLogin()

    def createBtn(self):
        self.reset()
        sm.current = "create"

    def reset(self):
        self.email.text = ""
        self.password.text = ""


class MainWindow(Screen):
    n = ObjectProperty(None)
    created = ObjectProperty(None)
    email = ObjectProperty(None)
    current = ""

    def logOut(self):
        sm.current = "login"

    def on_enter(self, *args):
        password, name, created = db.get_user(self.current)
        self.n.text = "Account Name: " + name
        self.email.text = "Email: " + self.current
        self.created.text = "Created On: " + created


class MainPage(Screen):
    def PoseCalculate(self):
        pose_classification()
        
    # label = MDLabel(text="Hello, World!", halign="center", valign="middle")
    
    # RectangleFlatButton(
    #     pos_hint={"x":0.3,"y":0.7}
    #     size_hint=0.4, 0.09
    #     text= "Start Session "
    #     font_size= 20)

    # RectangleFlatButton(
    #     pos_hint={"x":0.3,"y":0.55}
    #     size_hint=0.4, 0.09
    #     text="Session History"
    #     font_size= 20)

    # RectangleFlatButton(
    #     pos_hint={"x":0.3,"y":0.4}
    #     size_hint= 0.4, 0.09
    #     text= "Analysis"
    #     font_size= 20)

    # RectangleFlatButton(
    #     background_color=(0/255,150/255,255/255,1)
    #     pos_hint={"x":0.2,"y":0.09}
    #     size_hint= 0.2, 0.07
    #     text= "Logout"
    #     font_size= (root.width**2 + root.height**2) / 14**4)


class WindowManager(ScreenManager):
    pass


def invalidLogin():
    pop = Popup(title='Invalid Login',
                  content=Label(text='Invalid username or password.'),
                  size_hint=(None, None), size=(300, 200))
    pop.open()


def invalidForm():
    pop = Popup(title='Invalid Form',
                  content=Label(text='Please fill in all inputs with valid information.'),
                  size_hint=(None, None), size=(300, 200))

    pop.open()


def changeScreen(self,*args):
    sm.current = "login"
    return sm



kv = Builder.load_file("my.kv")

sm = WindowManager()
db = DataBase("users.txt")

screens = [SplashScreen(name="SplashScreen"), LoginWindow(name="login"), CreateAccountWindow(name="create"),MainPage(name="mainPage"),MainWindow(name="main")]
for screen in screens:
    sm.add_widget(screen)

sm.current = "SplashScreen"


class MyMainApp(App):
    def build(self):
        if sm.current == "SplashScreen":
             Clock.schedule_once(changeScreen, 5)
        return sm


if __name__ == "__main__":
    MyMainApp().run()
