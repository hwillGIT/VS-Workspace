package com.example;

import java.util.ArrayList;

public class RemoveEvens {
    //main method
    public static void main(String[] args) {
        int[] outputArray = deleteEvens(new int[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    //print the output array
        for (int i = 0; i < outputArray.length; i++) {
            System.out.print(outputArray[i] + " ");
        }
    }

    private static int[] deleteEvens(int[] arr) {
        int[] holdingArray = new int[arr.length];
        int j = 0;
        for (int i = 0; i < arr.length; i++) {
            if (arr[i] % 2 != 0) {
                holdingArray[i] = arr[i];
                j++;
            }
        }
        int[] outArray = new int[j];
        j=0;
        for (int i = 0; i < holdingArray.length; i++) {
            if (holdingArray[i] != 0) {
                outArray[j] = holdingArray[i];
                j++;
            }
        }
        return outArray;
    }
    }
       
