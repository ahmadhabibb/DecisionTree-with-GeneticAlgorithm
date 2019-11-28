from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from pandas import DataFrame
import pandas as pd
import numpy as np
import random
import math
import sys

# Ahmad Habib Fitriansyah
# Bandung, Indonesia
# 17 November 2019, 12:11 PM

def read_dataLatih():
    col_names = ['Suhu', 'Waktu', 'KondisiLangit', 'Kelembapan', 'Terbang/Tidak']
    file = pd.read_csv("E:/Learning Coding/AI/Tugas 2/data_latih.csv", header=None, names=col_names)
    output = file.iloc[:,:5]
    noTerbang = file.iloc[:,:4]
    terbang = file.iloc[:,4]

    return noTerbang, terbang

def read_dataUji():
    col_names = ['Suhu', 'Waktu', 'KondisiLangit', 'Kelembapan']
    file = pd.read_csv("E:/Learning Coding/AI/Tugas 2/data_uji.csv", header=None, names=col_names)
    outputDataUji = file.iloc[:,:4]

    return outputDataUji

def convertBinaryUji(outputDataUji):
    onehot_encoder = OneHotEncoder(sparse=False)
    onehot_encoded = onehot_encoder.fit_transform(outputDataUji)

    return onehot_encoded

def convertBinaryLatih(noTerbang, terbang):
    onehot_encoder = OneHotEncoder(sparse=False)
    onehot_encoded = onehot_encoder.fit_transform(noTerbang)
    datalist = []
    addData = []
    newData = []
    resultOfConvert = []
    datalist.append(onehot_encoded)
    for i in range(len(terbang)):
        if(terbang[i] == "tidak"):
            addData.append(0)
            resultOfConvert = np.append(datalist[0][i], [addData])
            newData.append(resultOfConvert)
        elif(terbang[i] == "ya"):
            addData.append(1)
            resultOfConvert = np.append(datalist[0][i], [addData])
            newData.append(resultOfConvert)
        addData = []
        
    return newData

def createIndividu(x):
    individu = []
    rand = random.randint(1,5)
    for i in range(x*rand):
        individu.append(random.randint(0, 1))
    return individu
    
def createPopulasi(x, y):
    populasi = []
    for i in range(x):
        populasi.append(createIndividu(y))
    return populasi

def countFitness(datalist, populasi):
    fitness = []
    for i in range(len(populasi)):
        total = 0
        a = int(len(populasi[i])/15)
        for j in range(len(datalist)):
            stat = False
            b = 0
            while(not stat and b < a and populasi[(b*15) + 14 == 1]):
                status = 'false'
                status2 = 'false'
                status3 = 'false'
                status4 = 'false'
                ket = []
                kebenaran = False
                for k in range(3):
                    temp = math.ceil(datalist[j][k])
                    if(populasi[i][(b*15)+k] == temp == 1):
                        status = 'true'
                ket.append(status)
                for k in range(3,7):
                    temp2 = math.ceil(datalist[j][k])
                    if(populasi[i][(b*15)+k] == temp2 == 1):
                        status2 = 'true'
                ket.append(status2)
                for k in range(7,11):
                    temp3 = math.ceil(datalist[j][k])
                    if(populasi[i][(b*15)+k] == temp3 == 1):
                        status3 = 'true'
                ket.append(status3)
                for k in range(11,14):
                    temp4 = math.ceil(datalist[j][k])
                    if(populasi[i][(b*15)+k] == temp4 == 1):
                        status4 = 'true'
                ket.append(status4)

                if (ket[0] == ket[1] == ket[2] == ket[3] == 'true'):
                    kebenaran = True
                if (not kebenaran and datalist[j][14] == 0):
                    total += 1
                    stat = True
                elif (kebenaran and datalist[j][14] == 1):
                    total += 1
                    stat = True
                b += 1
        fitness.append(total/len(datalist))
    return fitness

def hitungProbabilitas(x):
    prob = 0
    prob = int(x * 10)
    return prob

def randomParent(panjang):
    rand = random.randint(0,panjang-1)
    rand2 = random.randint(0,panjang-1)
    while(rand == rand2):
        rand2 = random.randint(0,panjang-1)

    return rand, rand2

def crossOver(parent,x,y):
    parent1 = []
    parent2 = []
    child1 = []
    child2 = []
    sisa = 0

    parent1.append(parent[x])
    parent2.append(parent[y])
    if(len(parent1[0]) < len(parent2[0]) or len(parent1[0]) == len(parent2[0])):
        tPotong1_parent1 = random.randint(0,len(parent1[0])-1)
        tPotong2_parent1 = random.randint(0,len(parent1[0])-1)
        while(tPotong1_parent1 > tPotong2_parent1 or tPotong1_parent1 == len(parent1[0])-1 or tPotong2_parent1 == 0 or tPotong1_parent1 == tPotong2_parent1):
            tPotong1_parent1 = random.randint(0,len(parent1[0])-1)
            tPotong2_parent1 = random.randint(0,len(parent1[0])-1)
        titik1 = tPotong1_parent1
        titik2 = tPotong2_parent1
    else:
        tPotong1_parent2 = random.randint(0,len(parent2[0])-1)
        tPotong2_parent2 = random.randint(0,len(parent2[0])-1)
        while(tPotong1_parent2 > tPotong2_parent2 or tPotong1_parent2 == len(parent2[0])-1 or tPotong2_parent2 == 0 or tPotong1_parent2 == tPotong2_parent2):
            tPotong1_parent2 = random.randint(0,len(parent2[0])-1)
            tPotong2_parent2 = random.randint(0,len(parent2[0])-1)
        titik1 = tPotong1_parent2
        titik2 = tPotong2_parent2
    
    selisih = titik2 - titik1
    panjangAwal1 = titik1
    panjangAkhir1 = titik2
    panjangAwal2 = titik1
    panjangAkhir2 = titik2
    if (selisih > 15):
        sisa = selisih % 15

    for i in range(0,selisih+1):
        child2.append(parent2[0][titik1])
        titik1 = titik1 + 1
    titik1 = panjangAwal1

    for j in range(panjangAwal1):
        child2.append(parent1[0][j])
    for k in range(panjangAkhir1+1, len(parent1[0])):
        child2.append(parent1[0][k])

    for o in range(panjangAkhir2+1, len(parent2[0])):
        child1.append(parent2[0][o])
    
    for n in range(panjangAwal2):
        child1.append(parent2[0][n])
    
    for m in range(0,selisih+1):
        child1.append(parent1[0][titik1])
        titik1 = titik1 + 1
    
    return child1, child2

def getParent(array_probability):
    getArray = random.randint(0,(len(array_probability))-1)
    return array_probability[getArray]

def mutation(child):
    compare = []
    for i in range(len(child)):
        rand = random.randint(0,1)
        compare.append(rand)
        
    for j in range(len(compare)):
        if(compare[j] == 1):
            if(child[j] == 1):
                child[j] = 0
            else:
                child[j] = 1

    return child

def compareFitness(parent, child, checkFitness, checkFitnessAnak):
    tempFitness = []
    newGen = []
    big = 0
    mark = 0
    for fit1 in range(len(checkFitness)):
        tempFitness.append(checkFitness[fit1])
        
    for fit2 in range(len(checkFitnessAnak)):
        tempFitness.append(checkFitnessAnak[fit2])

    pArrayFitness = len(tempFitness)
    for loop in range(len(tempFitness) // 2):
        for i in range(len(tempFitness)):
            if(big < tempFitness[i]):
                big = tempFitness[i]
    
        for j in range(len(tempFitness)):
            if (tempFitness[j] == big):
                mark = j
        batas = pArrayFitness // 2

        if (mark >= batas):
            newGen.append(child[mark-batas])
        else:
            newGen.append(parent[mark])
        del tempFitness[mark]
        big = 0
        mark = 0

    return newGen
    
def checkTerbang(bestKromosom, convertUji):
    decision = []
    for j in range(len(convertUji)):
        kebenaran = False
        b = 0
        for x in range(len(bestKromosom)//15):
            while(b < len(bestKromosom)/15 and (not kebenaran) and bestKromosom[(b*15) + 14] == 1):
                ket = []
                total = 0
                # kebenaran = False
                for m in range(15):
                    if(bestKromosom[(b*15) + m] == 1):
                        total+=1
                if(total != 15):
                    status = 'false'
                    status2 = 'false'
                    status3 = 'false'
                    status4 = 'false'

                    for k in range(3):
                        temp = math.ceil(convertUji[j][k])
                        if(bestKromosom[(b*15)+k] == temp):
                            status = 'true'
                    ket.append(status)
                    for k in range(3,7):
                        temp2 = math.ceil(convertUji[j][k])
                        if(bestKromosom[(b*15)+k] == temp2):
                            status2 = 'true'
                    ket.append(status2)
                    for k in range(7,11):
                        temp3 = math.ceil(convertUji[j][k])
                        if(bestKromosom[(b*15)+k] == temp3):
                            status3 = 'true'
                    ket.append(status3)
                    for k in range(11,14):
                        temp4 = math.ceil(convertUji[j][k])
                        if(bestKromosom[(b*15)+k] == temp4):
                            status4 = 'true'
                    ket.append(status4)

                    if (ket[0] == ket[1] == ket[2] == ket[3] == 'true'):
                        kebenaran = True
                b += 1
            b += 1
            
        if (kebenaran == True):
            decision.append("Ya")
        elif (kebenaran == False):
            decision.append("Tidak")
         
    return decision
    
def main():
    panjangPopulasi = 200
    panjangIndividu = 15
    jumGenerasi = 10

    data, terbang = read_dataLatih()
    dataUji = read_dataUji()
    print("--------------- Data Uji ---------------")
    print(dataUji)
    convertUji = convertBinaryUji(dataUji)
    convert = convertBinaryLatih(data, terbang)

    print()
    print("Loading...")
    for loop in range(jumGenerasi):
        populasi = []
        array_probability = []
        jumFitness = 0
        parent = []
        child = []
        tempChild1 = []
        tempChild2 = []
        newChild = []

        if (len(populasi) == 0):
            populasi = createPopulasi(panjangPopulasi, panjangIndividu)

        checkFitness = countFitness(convert, populasi)
        for k in range(len(checkFitness)):
            roundNum = round(checkFitness[k],1)
            countProbability = hitungProbabilitas(roundNum)
            for m in range(countProbability):
                array_probability.append(k)

        for n in range(panjangPopulasi):
            mark = getParent(array_probability)

            parent.append(populasi[mark])

        pParent = len(parent)
        for p in range(len(parent) // 2):
            x, y = randomParent(pParent)
            tempChild1, tempChild2 = crossOver(parent,x,y)
            child.append(tempChild1)
            child.append(tempChild2)
        for loopChild in range(len(child)):
            mutation(child[loopChild])

        checkFitnessAnak = countFitness(convert, child)

    # MASUKIN SEMUA FITNESS KE 1 ARRAY
        populasi = []
        populasi = compareFitness(parent, child, checkFitness, checkFitnessAnak)
        # print(loop)   
        gen = loop+1
        sys.stdout.write("\rGenerasi ke-%i" % gen)
        sys.stdout.flush()
    print()
    # print(print.replace("Loading..."))
    
    bestFitness = countFitness(convert, populasi)
    best = 0
    mark = 0
    for xy in range(len(bestFitness)):
        if(best < bestFitness[xy]):
            best = bestFitness[xy]
    for yz in range(len(bestFitness)):
        if (bestFitness[yz] == best):
            mark = yz
    print("Akurasi =",best*100,"%")
    bestKromosom = populasi[mark]
    print("Rule :",bestKromosom)

    hasilKeputusan = checkTerbang(bestKromosom, convertUji)
    # print(*hasilKeputusan,sep="\n")
    print()
    print("Silahkan cek hasilnya prediksi di hasil.csv")

    df = DataFrame(hasilKeputusan)
    export_csv = df.to_csv("E:/Learning Coding/AI/Tugas 2/hasil.csv", header=False, index=None)

if __name__ == "__main__":
    main()