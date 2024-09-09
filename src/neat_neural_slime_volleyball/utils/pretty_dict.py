class PrettyDict(dict):
    def __str__(self):
        return '\n'.join(str(value) for value in self.values())
    
    def __repr__(self):
        return self.__str__()