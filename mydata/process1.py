import pandas as pd
import numpy as np
df=pd.read_csv('file:///Users/liuxueling/PycharmProjects/python3pj/5001personal/mydata/train.csv',encoding='utf-8')

'''
data['学历'].drop_duplicates()
educationLevelDict={'博士':4,'硕士':3,'大学':2,'大专':1}
data['学历 Map']=data['学历'].map(educationLevelDict)

dummies=pandas.get_dummies(
data,columns=['genres'],
prefix=['Genre'],
prefix_sep='_',
dummy_na=False,
drop_first=False
)
'''



def get_Stype(DF,str):
    SID = 0
    Sdf = pd.DataFrame(columns=[str])
    for i in (range(N)):
        for j in (range(len(DF[i]))):
            Sdf.loc[SID] = DF[i][j]
            SID += 1
    Sdf = Sdf.drop_duplicates([str], keep="first")
    Sdf.index = range(0, Sdf.shape[0])
    return Sdf



def get_S(DF,Stype,str):
    print('There are',Stype.shape[0],'typrs of',str,'\n')
    S_df = pd.DataFrame(np.zeros((DF.shape[0],Stype.shape[0])),columns=Stype[str].tolist(),index = DF.index)
    #print(genre_df.shape)
    for i in range(DF.shape[0]):
        for j in range(len(DF[i])):
            for k in range(Stype.shape[0]):
                #print(DF[i][j])
                #print(genretype[k])
                if  DF[i][j] == Stype[str].loc[k]:
                    S_df[DF[i][j]].loc[i] = 1
               # print(genre_df[DF[i][j]])

    def addtag(x):
        return x + '_' + str
    col = list(map(addtag,Stype[str].tolist()))
    print('col',col)
    S_df.columns=col
    return S_df


#########################################################################
#########################################################################
new_df = df['genres'].str.split(',', n=-1)
print(new_df)
N = new_df.shape[0]

S='genre'
genre_type = get_Stype(new_df,S)
genre_type.genre.tolist()
genres = get_S(new_df,genre_type,S)
print(genres.shape[1])

#########################################################################
#########################################################################

new_df = df['categories'].str.split(',', n=-1)
print(new_df)
N = new_df.shape[0]

S='categories'
categories_type = get_Stype(new_df,S)
categories_type[S].tolist()
categories = get_S(new_df,categories_type,S)
print(categories)

#########################################################################
#########################################################################

new_df = df['tags'].str.split(',', n=-1)
print(new_df)
N = new_df.shape[0]

S='tags'
tags_type = get_Stype(new_df,S)
tags_type[S].tolist()
tags = get_S(new_df,tags_type,S)
print(tags)

#########################################################################
#########################################################################
new_df = pd.concat([df,genres,categories,tags], axis=1, join='outer', join_axes=None, ignore_index=False)
print('new df:\n',new_df)
writer = pd.ExcelWriter('/Users/liuxueling/PycharmProjects/python3pj/5001personal/dataprocessiing/processed data/nominal.xls',engine='xlsxwriter')
new_df.to_excel(writer,sheet_name='nominal')
writer.save()
