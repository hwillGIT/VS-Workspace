package com.example.demo;

import java.util.ArrayList;

public class MergeTwoArrays {
    public static void main(String[] args) {
 
    }
//find unique elements in ArrayList
public ArrayList<Integer> findUnique(ArrayList<Integer> list) {
    ArrayList<Integer> uniqueList = new ArrayList<Integer>();
    for (int i = 0; i < list.size(); i++) {
        if (!uniqueList.contains(list.get(i))) {
            uniqueList.add(list.get(i));
        }
    }
    return uniqueList;
}
}

    
