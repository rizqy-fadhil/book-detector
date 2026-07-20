from roboflow import Roboflow
rf = Roboflow(api_key="xIMpnLnHVjcXz34CDP6a")
project = rf.workspace("book-u06zw").project("here-is-the-book")
version = project.version(6)
dataset = version.download("yolov8")