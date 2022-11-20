import ipdb

class cwf_pdb_test:
    def __init__(self,name) -> None:
        self.test_name = name
        pass

    def getCwfTestName(self):
        return self.test_name

    

if __name__ == "__main__":
    a=cwf_pdb_test("cwf1")
    ipdb.set_trace()
    print(a)

