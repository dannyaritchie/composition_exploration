import numpy as np

class data_input():
    #class to read and store sample data from standardised spread sheet
    #needs manual check that spreadsheet is of correct format
    def __init__(self,file):
        #read in data
        self.samples=[]
        with open(file) as fp:
            lines=fp.readlines()
            count=0
            num_elements=None
            num_inputs=None
            num_knowns=None
            phase_field=None
            formulas=[]
            for line in lines:
                chunks=line.split(",")
                if count==0:
                    num_elements=int(chunks[3])
                    num_inputs=int(chunks[1])
                    num_knowns=int(chunks[5])
                if count==2:
                    self.phase_field=chunks[num_inputs:num_inputs+num_elements]
                    for i in range(num_knowns):
                        formulas.append(chunks[num_inputs+num_elements+i])
                if count>2:
                    pos=chunks[num_inputs:num_inputs+
                                                 num_elements]
                    pos=np.array([float(x) for x in pos])
                    weights=chunks[num_inputs+num_elements:num_inputs+
                                   num_elements+num_knowns]
                    weights=np.array([float(x) for x in weights])
                    sample=(pos,weights)
                    self.samples.append(sample)
                count+=1
        #convert formulas to standard format
        self.formulas=[]
        for i in formulas:
            chunks = i.split()
            f_stan=""
            for el in self.phase_field:
                f_stan+=el + " "
                try:
                    n=chunks.index(el)
                except ValueError:
                    f_stan += "0 "
                else:
                    try:
                        f_stan+= str(float(chunks[n+1])) + " "
                    except:
                        f_stan+= "1 "
            self.formulas.append(f_stan)

    def get_samples(self, normalise_weights=False):
        if not normalise_weights:
            return self.samples
        else:
            samples_n=[]
            for i in self.samples:
                w_n=100*i[1]/np.sum(i[1])
                samples_n.append((i[0],w_n))
            return samples_n

    def get_formulas(self):
        return self.formulas

    def get_phase_field(self):
        return self.phase_field



