# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 14:48:38 2023

@author: acer895395543
"""

import os
import csv
import numpy as np
import math
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('MOF_name',type=str)
parser.add_argument('a',type=float)
parser.add_argument('b',type=float)
parser.add_argument('c',type=float)
parser.add_argument('alpha',type=float)
parser.add_argument('beta',type=float)
parser.add_argument('gamma',type=float)

#get MOF name
args=parser.parse_args()
MOF_name=args.MOF_name
#get MOF unit cell info [a (Å),b (Å),c (Å),alpha (degree),beta (degree),gamma (degree)] 
a=args.a
b=args.b
c=args.c
alpha=args.alpha/180*math.pi
beta=args.beta/180*math.pi
gamma=args.gamma/180*math.pi


#resolution (Å)
resolution=0.5
#temperature (K)
temperature=298
#energy threshold (K)
Energy_threshold=-4500
#distance threshold (Å)
distance_threshold=3.5

#compute the energy grid
def get_energy_grid():
    #read energymap data
    work=os.getcwd()
    path=os.path.join(work,MOF_name+'.txt')
    f=open(path,'r',newline='')
    reader=csv.reader(f)
    Total_Table=[]
    for line in reader:
        Total_Table.append(line)      
    f.close()
    
    #get position array and energy array
    Total_Table=np.array(Total_Table).astype(float)
    Total_Table=np.array(sorted(Total_Table,key=lambda x:x[3],reverse=False))
    Position=Total_Table[:,[0,1,2]]
    Energy=Total_Table[:,3]
    
    
    #Project all the points into an unit cell
    x_new=np.array([a,0,0])
    y_new=np.array([b*np.cos(gamma),b*np.sin(gamma),0])
    z_new=np.array([c*np.cos(beta),c*np.cos(alpha),c*np.sqrt(np.abs(np.sin(beta)**2-np.cos(alpha)**2))])
    x_new=x_new.reshape(3,1)
    y_new=y_new.reshape(3,1)
    z_new=z_new.reshape(3,1)
    total=np.hstack((x_new,y_new,z_new))
    old2new=np.linalg.inv(total)
    x_pron_coeff=np.dot(old2new,np.array([1,0,0]))
    y_pron_coeff=np.dot(old2new,np.array([0,1,0]))
    z_pron_coeff=np.dot(old2new,np.array([0,0,1]))
    x_pron_coeff=x_pron_coeff.reshape(1,3)
    y_pron_coeff=y_pron_coeff.reshape(1,3)
    z_pron_coeff=z_pron_coeff.reshape(1,3)
    #Get Transform matrix  (Cartesian to Fractional)
    Transform_factor=np.vstack((x_pron_coeff,y_pron_coeff,z_pron_coeff))
    #Get Transform matrix  (Fractional to Cartesian)
    Transform=np.transpose(total)
    
    New_position=np.dot(Position,Transform_factor)
    New_position=New_position%1
    
    
    #Cut the unit cell by the resoultion setting
    a_cut_number=math.ceil(a/resolution)
    b_cut_number=math.ceil(b/resolution)
    c_cut_number=math.ceil(c/resolution)
    a_resolution=1/a_cut_number
    b_resolution=1/b_cut_number
    c_resolution=1/c_cut_number
    
    #Find the strongest energy in each grid
    Have_data=np.zeros((a_cut_number,b_cut_number,c_cut_number)).astype(int)
    Max_energy=np.zeros((a_cut_number,b_cut_number,c_cut_number)).astype(float)
    for i in range(len(New_position)):
        check_position=np.round((New_position[i]//np.array([a_resolution,b_resolution,c_resolution]))%np.array([a_cut_number,b_cut_number,c_cut_number])).astype(int)
        if Have_data[tuple(check_position)]==0:
            Max_energy[tuple(check_position)]=Energy[i]
            Have_data[tuple(check_position)]=1
    
    #Compute the energy grid
    Accumuate_energy=np.zeros((a_cut_number,b_cut_number,c_cut_number)).astype(float)
    Weight=np.zeros((a_cut_number,b_cut_number,c_cut_number)).astype(float)
    for i in range(len(New_position)):
        check_position=np.round((New_position[i]//np.array([a_resolution,b_resolution,c_resolution]))%np.array([a_cut_number,b_cut_number,c_cut_number])).astype(int)
        Accumuate_energy[tuple(check_position)]+=Energy[i]*np.exp((-Energy[i]+Max_energy[tuple(check_position)])/temperature)
        Weight[tuple(check_position)]+=np.exp((-Energy[i]+Max_energy[tuple(check_position)])/temperature)
    Avg_energy=np.divide(Accumuate_energy, Weight, out=np.zeros_like(Weight), where=Weight!=0.0)
    Final=[]
    take=np.where(Avg_energy!=0.0)
    for i in range(len(take[0])):
        get_position=tuple([take[0][i],take[1][i],take[2][i]])
        Final.append([take[0][i]*a_resolution,take[1][i]*b_resolution,take[2][i]*c_resolution,Avg_energy[get_position]])
    Final=np.array(Final)
    
    
    #get the adsorption sites by the energy threshold
    Final=Final[Final[:,3]<Energy_threshold]

    
    return Final , Transform

#Clustering routine
def clustering_routine(Table,Transform):
    #get position data
    Positions=Table[:,[0,1,2]]
    
    #extend the unit cell by 3.5 Å and get the size of the supercell
    adjust=np.sin(gamma)
    box_a1=np.ceil(100*distance_threshold/adjust/a)/100
    box_b1=np.ceil(100*distance_threshold/adjust/b)/100
    adjust=np.sin(beta)
    box_a2=np.ceil(100*distance_threshold/adjust/a)/100
    box_c1=np.ceil(100*distance_threshold/adjust/c)/100
    adjust=np.sin(alpha)
    box_b2=np.ceil(100*distance_threshold/adjust/b)/100
    box_c2=np.ceil(100*distance_threshold/adjust/c)/100
    box_a=np.max((box_a1,box_a2))
    box_b=np.max((box_b1,box_b2))
    box_c=np.max((box_c1,box_c2))
    cut_criteria=np.array([box_a,box_b,box_c])
    Extend_N=np.ceil(np.max(cut_criteria)).astype(int)
    
    #Get all the point of the supercell
    Extend_point=[]
    Extend_point.extend(Positions.tolist())
    for a_extend in range(-Extend_N,Extend_N+1):
        for b_extend in range(-Extend_N,Extend_N+1):
            for c_extend in range(-Extend_N,Extend_N+1):
                if a_extend!=0 or b_extend!=0 or c_extend!=0:
                    new_extend_point=(Positions+np.array([a_extend,b_extend,c_extend])).tolist()
                    Extend_point.extend(new_extend_point)          
    Extend_point=np.array(Extend_point)
    
    
    #Clustering routine
    cluster_ID=0
    cluster_result=np.array([0 for i in range(Positions.shape[0])])
    not_search_index=np.array([i for i in range(Extend_point.shape[0])])
    while len(not_search_index)>0:
        include_index_type=not_search_index[0]%len(Positions)
        this_cluster_index=[include_index_type]
        connect_result=[include_index_type]
        not_search_index=not_search_index[~np.isin(not_search_index%len(Positions),include_index_type)]
        while len(connect_result)>0:
            new_connect_index=[]
            for point_index in connect_result:
                start_point=Extend_point[point_index]
                Upper_bound=start_point+cut_criteria
                Lower_bound=start_point-cut_criteria
                if len(not_search_index)>0:
                    Other_point=Extend_point[not_search_index]
                    check_upper=Other_point<=Upper_bound
                    check_lower=Other_point>=Lower_bound
                    total_check=np.all(np.hstack((check_upper,check_lower)),axis=1)
                    possible_index=not_search_index[total_check]
                    Destination=Other_point[total_check]
                    diff=np.dot(start_point-Destination,Transform)
                    inner_distance=np.sqrt(np.sum(np.power(diff,2),axis=1))
                    include_index_type=possible_index[inner_distance<=distance_threshold]%len(Positions)
                    this_cluster_index.extend(include_index_type.tolist())
                    new_connect_index.extend(include_index_type.tolist())
                    not_search_index=not_search_index[~np.isin(not_search_index%len(Positions),include_index_type)]
                else:
                    break
            connect_result=new_connect_index[:]
            
        cluster_result[this_cluster_index]=cluster_ID
        cluster_ID+=1
        
    #Merge the clustering result in the position array and the energy array
    cluster_result=cluster_result.reshape(cluster_result.shape[0],1)
    Point_Energy_cluster=np.hstack((Table,cluster_result))
    
    return Point_Energy_cluster , cut_criteria

#channel routine
def channel_routine(Table,cut_criteria,Transform):
    #find channel (0 = no, 1 = yes)
    find_channel=0
    #get all clustere ID
    cluster_ID=set(Table[:,4])
    for ID in cluster_ID:
        #get the size of the supercell containing one continuous cluster and the neighborhood of the cluster
        Points=Table[Table[:,4]==ID][:,[0,1,2]]
        Start_point=Points[0]
        disrete_diff=Start_point.reshape(1,3)-Points
        dicrete_diff_array=np.dot(disrete_diff,Transform)
        cut_inner_distance=np.max(np.sqrt(np.sum(np.power(dicrete_diff_array,2),axis=1)))
        radius=cut_inner_distance+distance_threshold
        #ab & gamma
        x1=Start_point[0]*a
        y1=Start_point[1]*b
        adjust=np.sin(gamma)
        n1=np.ceil((radius/adjust+x1-a)/a)
        n2=np.ceil((radius/adjust-x1)/a)
        n3=np.ceil((radius/adjust+y1-b)/b)
        n4=np.ceil((radius/adjust-y1)/b)
        #ac & beta
        x1=Start_point[0]*a
        z1=Start_point[2]*c
        adjust=np.sin(beta)
        n5=np.ceil((radius/adjust+x1-a)/a)
        n6=np.ceil((radius/adjust-x1)/a)
        n7=np.ceil((radius/adjust+z1-c)/c)
        n8=np.ceil((radius/adjust-z1)/c)
        #bc & alpha
        y1=Start_point[1]*b
        z1=Start_point[2]*c
        adjust=np.sin(alpha)
        n9=np.ceil((radius/adjust+y1-b)/b)
        n10=np.ceil((radius/adjust-y1)/b)
        n11=np.ceil((radius/adjust+z1-c)/c)
        n12=np.ceil((radius/adjust-z1)/c)
        #calculate Extend_N
        a_plus=np.round(np.max((n1,n5),axis=0)).astype(int)
        a_minus=np.round(-np.max((n2,n6),axis=0)).astype(int)
        b_plus=np.round(np.max((n3,n9),axis=0)).astype(int)
        b_minus=np.round(-np.max((n4,n10),axis=0)).astype(int)
        c_plus=np.round(np.max((n7,n11),axis=0)).astype(int)
        c_minus=np.round(-np.max((n8,n12),axis=0)).astype(int)
        Extend_N=np.max([a_plus,-a_minus,b_plus,-b_minus,c_plus,-c_minus])
      
        #Get all possible points
        Extend_point=[]
        Extend_point.extend(Points.tolist())
        for a_extend in range(-Extend_N,Extend_N+1):
            for b_extend in range(-Extend_N,Extend_N+1):
                for c_extend in range(-Extend_N,Extend_N+1):
                    if a_extend!=0 or b_extend!=0 or c_extend!=0:
                        new_extend_point=(Points+np.array([a_extend,b_extend,c_extend])).tolist()
                        Extend_point.extend(new_extend_point)      
        Extend_point=np.array(Extend_point)
        
        
        #get a continuous cluster by clustering routine
        not_search_index=np.array([i for i in range(Extend_point.shape[0])]).astype(int)
        this_cluster_index=[0]
        connect_result=[0]
        not_search_index=not_search_index[~np.isin(not_search_index%len(Points),0)]
        while len(connect_result)>0:
            new_connect_index=[]
            for point_index in connect_result:
                start_point=Extend_point[point_index]
                Upper_bound=start_point+cut_criteria
                Lower_bound=start_point-cut_criteria
                if len(not_search_index)>0:
                    Other_point=Extend_point[not_search_index]
                    check_upper=Other_point<=Upper_bound
                    check_lower=Other_point>=Lower_bound
                    total_check=np.all(np.hstack((check_upper,check_lower)),axis=1)
                    possible_index=not_search_index[total_check]
                    Destination=Other_point[total_check]
                    diff=np.dot(start_point-Destination,Transform)
                    inner_distance=np.sqrt(np.sum(np.power(diff,2),axis=1))
                    get_qualified_index_in_possible_index=np.where(inner_distance<=distance_threshold)[0]
                    Mini_index_dict={}
                    for catch_index in get_qualified_index_in_possible_index:
                        specific_point_index=possible_index[catch_index]
                        if specific_point_index%len(Points) not in Mini_index_dict.keys():
                            Mini_index_dict[specific_point_index%len(Points)]=[specific_point_index,inner_distance[catch_index]]
                        else:
                            if inner_distance[catch_index]<Mini_index_dict[specific_point_index%len(Points)][1]:
                                Mini_index_dict[specific_point_index%len(Points)]=[specific_point_index,inner_distance[catch_index]]
                    New_node=[]
                    for key in Mini_index_dict.keys():
                        New_node.append(Mini_index_dict[key][0])
                    this_cluster_index.extend(New_node)
                    new_connect_index.extend(New_node)
                    include_index_type=np.array(New_node)%len(Points)
                    not_search_index=not_search_index[~np.isin(not_search_index%len(Points),include_index_type)]
                else:
                    break
            connect_result=new_connect_index[:]               
        
        
        #search points (in the neighborhood, 3.5 A) connected to the cluster
        connect_node_index=[]
        Remain_index=np.array([i for i in range(Extend_point.shape[0])]).astype(int)
        Remain_index=Remain_index[~np.isin(Remain_index,this_cluster_index)]
        Upper_bound=np.max(Extend_point[this_cluster_index],axis=0)+cut_criteria
        Lower_bound=np.min(Extend_point[this_cluster_index],axis=0)-cut_criteria
        Other_point=Extend_point[Remain_index]
        check_upper=Other_point<=Upper_bound
        check_lower=Other_point>=Lower_bound
        total_check=np.all(np.hstack((check_upper,check_lower)),axis=1)
        not_search_index=Remain_index[total_check]

        for this_index in this_cluster_index:
            start_point=Extend_point[this_index]
            Upper_bound=start_point+cut_criteria
            Lower_bound=start_point-cut_criteria
            Other_point=Extend_point[not_search_index]
            check_upper=Other_point<=Upper_bound
            check_lower=Other_point>=Lower_bound
            total_check=np.all(np.hstack((check_upper,check_lower)),axis=1)
            if np.sum(total_check)>0:
                possible_index=not_search_index[total_check]
                Destination=Other_point[total_check]
                diff=np.dot(start_point-Destination,Transform)
                inner_distance=np.sqrt(np.sum(np.power(diff,2),axis=1))
                include_index_type=possible_index[inner_distance<=distance_threshold]
                connect_node_index.extend(include_index_type.tolist())
                not_search_index=not_search_index[~np.isin(not_search_index,include_index_type)]
        
        #if the cluster can connect to at least one point in the neighborhood, then a CAC forms
        if len(connect_node_index)>0:
            find_channel=1
            break
        
    return find_channel
    
        

#get energy grid & the transform matrix (Fractional coordinate 2 Cartesian coordinate)
Energy_grid , Frac2Real = get_energy_grid()

#if there are adsorption sites in the unit cell
if len(Energy_grid)>0:
    #clustering routine & neighborhood
    Clustering_result , neighbor=clustering_routine(Energy_grid,Frac2Real)
    #channel routine
    detect_channel=channel_routine(Clustering_result,neighbor,Frac2Real)
else:
    #detect channel (0 = no, 1 = yes)
    detect_channel=0     
        


#write report
work=os.getcwd()
path_report=os.path.join(work,'CAC_report.csv')
f_report=open(path_report,'w',newline='')
writer=csv.writer(f_report)
writer.writerow(['MOF_name','detect CAC'])
if detect_channel==0:
    writer.writerow([MOF_name,'no'])
elif detect_channel==1:
    writer.writerow([MOF_name,'yes'])
f_report.close()


    
        
        






