
from roboflow import Roboflow
rf = Roboflow(api_key="3NXPCW5B6zKt8edO3ci8")
project = rf.workspace("new-workspace-w8cwp").project("faced-wwt2a")
version = project.version(3)
                
dataset = version.download("yolov7",location= '/mnt/md0/projects/nguyendai-footage/roboflow/close_face_new')
                
