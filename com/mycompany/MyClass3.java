package com.mycompany;
public class MyClass3 {
    //swap 2 elements of array
    public static void swap(int[] arr, int i, int j) {
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }   
    //find index of lowest element in array
    public static int findMinIndex(int[] arr, int startIndex) {
        int minIndex = startIndex;
        for (int i = startIndex + 1; i < arr.length; i++) {
            if (arr[i] < arr[minIndex]) {
                minIndex = i;
            }
        }
        return minIndex;
    }
    //sort elements using selectionSort
    public static void selectionSort(int[] arr) {
        int minIndex;
        for (int i = 0; i < arr.length - 1; i++) {
            minIndex = findMinIndex(arr, i);
            swapElements(arr, minIndex, i);
        }
    }
    private static void swapElements(int[] arr, int i, int j) {
        int temp;
        temp = arr[j];
        arr[j] = arr[i];
        arr[i] = temp;
    }
 
    public static void main(String[] args) {
        int[] arr = {1, 26, 366,56, 5, 6, 23, 8, 23, 180};
        selectionSort(arr);
        for (int element : arr) {
            System.out.println(element);
        }
    }

}
