from Face import Face

class RecognizedFace:
    """
    Represents a recognized face with a name, distance, and associated face object.
    """

    def __init__(self, name, distance, face: Face):
        self.name = name
        self.distance = distance
        self.face = face

    
