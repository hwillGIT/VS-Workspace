package patterns;

public class MyClass4 {

   public static void main(String[] args) {
      //find subarray containing duplicates in string array of size 10
      String[] str = {"a", "b", "e", "d", "e", "f", "b", "h", "d", "j"};

      int start = 0;
      int end = 0;
      int count = 0;
      int maxCount = 0;
      int maxStart = 0;
      int maxEnd = 0;
      
      for (int i = 0; i < str.length; i++) {
         for (int j = i + 1; j < str.length; j++) {
            if (str[i].equals(str[j])) {
               count++;
               if (count > maxCount) {
                  maxCount = count;
                  maxStart = start;
                  maxEnd = end;
               }
            }
         }
         count = 0;
         start = i + 1;
      }  //end for
      System.out.println("Subarray containing duplicates is: ");
      for (int i = maxStart; i <= maxEnd; i++) {
         System.out.print(str[i] + " ");
      }


   

   }

   

}



