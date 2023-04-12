import com.sun.scenario.effect.Merge;

import java.util.*;

public class Solution {
    public static void main(String[] args) {
//        System.out.println(countWords(4, 1));
//        int[] nums = {10,3,5,7};
//        System.out.println(smallestMax(nums));
//        System.out.println(countVowelSubstrings("aeioaexaaeuiou"));
//        int[][] grid = {{4,5}, {6,7}};
//        int maxSum = 2;
//        System.out.println(largestSubgrid(grid, maxSum));
//        System.out.println(product("axbawbaseksqke"));
//        System.out.println("Expected: 1, actual: " + getMinCost(new int[] {1,1,3,4}, new int[] {3,1,2,3}));
//        System.out.println("Expected: 3, actual: " + getMinCost(new int[] {1,2,3,2}, new int[] {1,2,3,2}));
//        System.out.println("Expected: 4, actual: " + getMinCost(new int[] {2,3,4,2}, new int[] {1,1,1,1}));
//        System.out.println("Expected: 4, actual: " + getMinCost(new int[] {2,3,4,5}, new int[] {1,1,5,3}));
//        int[][] nums = {{5,3,3}, {3,4,6},{2,4,1},{2,1,6}};
//        System.out.println(getMaxNetVulnerability(nums));
//        int[] nums = {5,2,13,10};
//        System.out.println(getMaxBarrier(nums, 8));
    }

    //1. String Pattern
    // Power function to calculate
    // long powers with mod
    public static int power(int x, int y, int p) {
        int res = 1;
        x = x % p;
        if (x == 0) return 0;
        while (y > 0) {
            if ((y & 1) != 0) res = (res * x) % p;
            y = y >> 1;
            x = (x * x) % p;
        }
        return res;
    }

    // Function for finding number of ways to
    // create string with length N and atmost
    // K contiguous vowels
    public static int countWords(int N, int K) {
        int i, j;
        int MOD = 1000000007;
        // Array dp to store number of ways
        long[][] dp = new long[N + 1][K + 1] ;
        long sum = 1;
        for(i = 1; i <= N; i++) {
            // dp[i][0] = (dp[i-1][0]+dp[i-1][1]..dp[i-1][k])*21
            dp[i][0] = sum * 21;
            dp[i][0] %= MOD;
            // Now setting sum to be dp[i][0]
            sum = dp[i][0];
            for(j = 1; j <= K; j++) {
                // If j>i, no ways are possible to create
                // a string with length i and vowel j
                if(j > i) dp[i][j] = 0;
                else if (j == i) {
                    // If j = i all the character should
                    // be vowel
                    dp[i][j] = power(5, i, MOD);
                } else {
                    // dp[i][j] relation with dp[i-1][j-1]
                    dp[i][j] = dp[i - 1][j - 1] * 5;
                }
                dp[i][j] %= MOD;
                // Adding dp[i][j] in the sum
                sum += dp[i][j];
                sum %= MOD;
            }
        }
        return (int)sum;
    }

//    2. String Formation
    String[] words;
    String target;
    Integer[][] memo;
    int m, n;
    int[][] charAtIndexCnt;
    public int numWays(String[] words, String target) {
        this.words = words; this.target = target;
        m = words[0].length(); n = target.length();
        memo = new Integer[m][n];
        charAtIndexCnt = new int[128][m];

        for (String word : words)
            for (int i = 0; i < word.length(); i++)
                charAtIndexCnt[word.charAt(i)][i] += 1; // Count the number of character `c` at index `i` of all words
        return dp(0, 0);
    }

    public int dp(int k, int i) {
        if (i == n)  // Formed a valid target
            return 1;
        if (k == m)  // Reached to length of words[x] but don't found any result
            return 0;
        if (memo[k][i] != null) return memo[k][i];
        char c = target.charAt(i);
        long ans = dp(k + 1, i);  // Skip k_th index of words
        if (charAtIndexCnt[c][k] > 0) { // Take k_th index of words if found character `c` at index k_th
            ans += (long) dp(k + 1, i + 1) * charAtIndexCnt[c][k];
            ans %= 1_000_000_007;
        }
        return memo[k][i] = (int) ans;
    }
//    3. Maximize Array Value
    public static int smallestMax(int[] nums){
        int n = nums.length;
        // we can not reduce the 1st number, so it is our lower bound;
        int lo = nums[0];
        int hi = getMax(nums);
        int minmax = lo;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (canAchieve(nums, mid)) {
                minmax = mid;
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        return minmax;
    }
    private static boolean canAchieve(int[] nums, int max){
        int diff = 0;
        for (int i = nums.length - 1; i >= 0; i--){
            diff = Math.max(nums[i] + diff - max, 0);
        }
        return diff == 0;
    }

    private static int getMax(int[] nums) {
        int max = nums[0];
        for(int n: nums){
            max = Math.max(max, n);
        }
        return max;
    }
//    4. Perfect Pairs
    public static int numberOfPair(int[] arr){
    for (int i = 0; i < arr.length; i++){
        if (arr[i] < 0) arr[i] = -arr[i];
    }
    Arrays.sort(arr);
    int res = 0;
    for (int i = 0; i < arr.length; i++){
        int idx = bs(arr, arr[i] * 2);
        res += Math.max(0, idx - i);
    }
    return res;
}

    // find the int x that arr[x] <= n < arr[x + 1]
    public static int bs(int[] arr, int n){
        int sz = arr.length;
        if (n <= arr[0]) return 0;
        if (arr[sz - 1] <= n) return sz - 1;

        int lo = 0;
        int hi = sz - 1;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (n < arr[mid]) {
                hi = mid - 1;
            } else if (arr[mid] <= n && n < arr[mid + 1]) {
                return mid;
            } else {
                lo = mid - 1;
            }
        }
        return hi;
    }
//    5. VowelSubstring
    public static int countVowelSubstrings(String word) {
        int i = 0, n = word.length();
        int res = 0;
        while (i < n) {
            if (isVowel(word.charAt(i))) {
                int j = i + 1;
                while (j < n && isVowel(word.charAt(j))) {
                    j++;
                }
                res += helper(word, i, j);
                i = j + 1;
            } else {
                i++;
            }
        }
        return res;
    }

    private static boolean isVowel(char c) {
        return "aeiou".indexOf(c) >= 0;
    }

    private static int helper(String s, int left, int right) {
        if (right - left < 5) return 0;
        // 用滑动窗口统计 符合条件的子字符串 数量
        Map<Character, Integer> count = new HashMap<>();
        int res = 0;
        int i = left;
        while (true) {
            while (i < right && count.size() < 5) {
                count.put(s.charAt(i), count.getOrDefault(s.charAt(i), 0) + 1);
                i++;
            }

            if (count.size() < 5) break;
            // s[left:i - 1] 包含五种元音字符，s[i:right - 1]长度为 right - i
            res += right - i + 1;
            count.put(s.charAt(left), count.get(s.charAt(left)) - 1);
            if (count.get(s.charAt(left)) == 0) {
                count.remove(s.charAt(left));
            }
            left++;
        }
        return res;
    }
//    6. LargestSubgrid
    static List<List<Integer>> preSum = new ArrayList<>();

    public static int getSum(int row, int col){
        if(row < 0 || col < 0) return 0;
        return preSum.get(row).get(col);
    }

    public static int sumRegion(int row1, int col1, int row2, int col2) {
        return getSum(row2,col2) - getSum(row1-1,col2) - getSum(row2,col1-1) + getSum(row1-1,col1-1);
    }

    public static int largestSubgrid(int[][] grid, int k) {
        int n = grid.length;
        preSum = new ArrayList<>();
        for(int i=0; i<n; i++){
            preSum.add(new ArrayList<Integer>());
            for(int j=0; j<n; j++){
                preSum.get(i).add(getSum(i-1,j) + getSum(i,j-1) - getSum(i-1,j-1) + grid[i][j]);
            }
        }
        int low = 0, high = n, ans = 0;
        while(low <= high){
            int mid = low + (high - low)/2;
            if(mid == 0) return 0;
            boolean stop = false;
            for(int i=mid-1; i<n && !stop; i++){
                for(int j=mid-1; j<n && !stop; j++){
                    int subSum = sumRegion(i-mid+1,j-mid+1,i,j);
                    if(subSum > k) stop = true;
                }
            }
            if(stop) high = mid - 1;
            else{
                ans = mid;
                low = mid + 1;
            }
        }
        return ans;
    }
//    7.Palindromic Sequence
    public static int product (String s) {
        int length = s.length();
        int[][] dp = new int[length][length];
        // stage one: from bottom to up, obtain the palindromic subsequence numbers in each substring
        for (int i = length - 1; i >= 0; i--) {
            dp[i][i] = 1;
            for (int j = i+1; j < length; j++) {
                if (s.charAt(i) == s.charAt(j)) {
                    dp[i][j] = dp[i+1][j-1] + 2;
                } else {
                    dp[i][j] = Math.max(dp[i+1][j], dp[i][j-1]);
                }
            }
        }
        // second stage: traverse the matrix, update maxProduct in each iteration
        // outer loop: traverse string s from i to length
        // inner loop: divide current s[i : length - 1] by j to left and right part, then obtain current product by multiply
        // palindromic numbers in s[i : j] and s[j : length - 1], which correspond to dp[i][j] and dp[j][length - 1], also update maxProduct
        int maxProduct = 0;
        for (int i = 0; i < length; i++) {
            for (int j = 0; j < length - 1; j++) {
                maxProduct = Math.max(maxProduct, dp[i][j] * dp[j + 1][length - 1]);
            }
        }
        return maxProduct;
    }
    static char[] ss;
    static int len;
    public static int maxProduct(String s) {
        len = s.length();
        int mask = (1 << len) - 1;
        ss = s.toCharArray();
        int res = 0;
        for(int i = 1; i <= mask; i++) {
            int x = getLen(i);
            int y = getLen(mask ^ i);
            res = Math.max(res, x * y);
        }
        return res;
    }
    public static int getLen(int state) {
        if(state == 0) return 0;
        StringBuilder sb = new StringBuilder();
        for(int i = len - 1; i >= 0; i--) {
            if(((state >> i) & 1) == 1) sb.append(ss[i]);
        }
        return maxLen(sb.toString());
    }
    public static int maxLen(String s) {
        int nn = s.length();
        int[][] dp = new int[nn][nn];
        for(int j = 0; j < nn; j++) {
            dp[j][j] = 1;
            for(int i = j - 1; i >= 0; --i) {
                if(s.charAt(i) == s.charAt(j)) dp[i][j] = 2 + dp[i + 1][j - 1];
                else dp[i][j] = Math.max(dp[i + 1][j], dp[i][j - 1]);
            }
        }
        return dp[0][nn - 1];
    }
//    8. TaskScheduling
    public static final long INF = Long.MAX_VALUE / 10;
        private static long getMinCost(int[] cost, int[] time) {
            assert cost.length > 0;
            assert cost.length == time.length;
            Map<Integer,Long>[] memo = new Map[cost.length];
            for (int i = 0; i < cost.length; i++) memo[i] = new HashMap<>();
            return dfs(0, cost, 0, time, memo);
        }

    public static long dfs(int i, int[] cost, int timeSoFar, int[] time, Map<Integer,Long>[] memo) {
        int n = cost.length;
        if (i == n) return timeSoFar >= 0 ? 0 : INF;
        if (timeSoFar >= n - i) return 0;
        if (memo[i].containsKey(timeSoFar)) return memo[i].get(timeSoFar);
        long resPaid = cost[i] + dfs(i+1, cost, timeSoFar + time[i], time, memo); // paid server
        long resFree = dfs(i+1, cost, timeSoFar - 1, time, memo); // free server
        memo[i].put(timeSoFar, Math.min(resPaid, resFree));
        return memo[i].get(timeSoFar);
    }
//    9 ServerSelection
    public static int getMaxNetVulnerability(int[][] vulnerability){
            int n = vulnerability.length, m = vulnerability[0].length;
            int[] row = new int[n];
            int maxVal = Integer.MIN_VALUE;

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                maxVal = Math.max(maxVal, vulnerability[i][j]);
            }
        }
        int low = 0;
        int high = maxVal;

        int ans = 0;
        while (low <= high) {
            int mid = low + (high - low) / 2;

            boolean flag = false;
            boolean flag1 = false;
            Arrays.fill(row,0);
            for (int j = 0; j < m; j++) {
                boolean found = false;
                for (int i = 0; i < n; i++) {
                    if (vulnerability[i][j] >= mid) {
                        found = true;

                        row[i]++;
                        if(row[i]>1)
                        {
                            flag1=true;
                            break;
                        }
                    }
                }
                if (!found) {
                    flag = true;
                    break;
                }
            }

            if (flag) {
                high = mid - 1;
            } else if(flag1){

                low = mid + 1;
                ans = mid;
            }
            else {
                high = mid - 1;
            }
        }
        return ans;
    }
//    10 Array reduction
    public static int getMex(int[] a, int i, int j) {
        int mex = 0;
        Set<Integer> seen = new HashSet<>();
        for (int k = i; k <= j; k++) {
            int el = a[k];
            seen.add(el);
            while (seen.contains(mex)) mex++;
        }

        return mex;
    }

    public static List<Integer> mexReduction(int[] a) {
        int n = a.length;
        int mex = getMex(a, 0, n - 1);
        Map<Integer, Integer> ct = new HashMap<>();
        for (int el : a) {
            ct.put(el, ct.getOrDefault(el, 0) + 1);
        }

        List<Integer> ret = new ArrayList<>();

        int i = 0;
        while (i < n) {
            int curMex = 0;
            Set<Integer> seen = new HashSet<>();
            while (curMex != mex) {
                int el = a[i];
                ct.put(el, ct.get(el) - 1);
                seen.add(el);

                while (seen.contains(curMex)) {
                    curMex++;
                }
                i++;
            }
            ret.add(curMex);

            if (curMex == 0) {
                i++;
            }

            int nextMex = 0;
            while (ct.containsKey(nextMex) && ct.get(nextMex) > 0) {
                nextMex++;
            }
            mex = nextMex;
        }

        return ret;
    }
//    11.Merge Interval
    public static int[][] merge(int[][] intervals) {
        Arrays.sort(intervals, (a, b)->a[0]-b[0]);
        int n = intervals.length;
        List<int[]> list = new ArrayList<>();
        list.add(intervals[0]);
        for(int i = 1; i < n; i++){
            int[] cur = intervals[i];
            int[] prev = list.get(list.size()-1);
            if(cur[0] > prev[1]) list.add(cur);
            else{
                prev[0] = Math.min(prev[0], cur[0]);
                prev[1] = Math.max(prev[1], cur[1]);
                list.set(list.size()-1, prev);
            }
        }
        return list.toArray(new int[list.size()][]);
    }
//    12.Best Interval / Minimum Interval to Include Each Query
    public static int[] minInterval(int[][] A, int[] queries) {
        TreeMap<Integer, Integer> pq = new TreeMap<>();
        HashMap<Integer, Integer> res = new HashMap<>();
        int i = 0, n = A.length, m = queries.length;
        int[] Q = queries.clone(), res2 = new int[m];
        Arrays.sort(A, (a, b) -> Integer.compare(a[0] , b[0]));
        Arrays.sort(Q);
        for (int q : Q) {
            while (i < n && A[i][0] <= q) {
                int l = A[i][0], r = A[i++][1];
                pq.put(r - l + 1, r);
            }
            while (!pq.isEmpty() && pq.firstEntry().getValue() < q)
                pq.pollFirstEntry();
            res.put(q, pq.isEmpty() ? -1 : pq.firstKey());
        }
        i = 0;
        for (int q : queries)
            res2[i++] = res.get(q);
        return res2;
    }
//    13. Grid Land / Kth Smallest Instructions
    public static String solve(int x, int y, int k) {
        int R = y + 1, C = x + 1; // x,y is column and row index,
        int p[][] = new int[R][C]; // p[i][j] keep sum of paths to (x,y) from (j,i)
        p[R - 1][C - 1] = 1;
        for (int i = R - 1; i >= 0; i--) {
            for (int j = C - 1; j >= 0; j--) {
                if (i == R - 1 && j == C - 1) continue;
                int r = (i + 1) == R ? 0 : p[i + 1][j];
                int d = (j + 1) == C ? 0 : p[i][j + 1];
                p[i][j] = r + d;
            }
        }
        StringBuilder a = new StringBuilder();
        int i = 0, j = 0;
        k++;
        while (true) {
            int right = (j + 1) == C ? 0 : p[i][j + 1];
            if (k <= right) {
                a.append("H");
                j++;
            } else {
                a.append("V");
                i++;
                k = k - right;
            }
            if (i == y && j == x) break;
        }
        return a.toString();
    }
//    14. CrossTheThreshold
    public static int getMaxBarrier(int[] initialEnergy, int th) {
        int maxEnergy = Arrays.stream(initialEnergy).max().getAsInt();
        int left = 0;
        int right = maxEnergy;
        while (left <= right) {
            int mid = (right - left) / 2 + left;
            int sum_m = getSum(initialEnergy, mid);
            if (sum_m == th) {
                return mid;
            } else if (sum_m > th) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return right;
    }

    public static int getSum(int[] initialEnergy, int barrier) {
        int ans = 0;
        for (int i : initialEnergy) {
            if (i - barrier > 0) {
                ans += i - barrier;
            }
        }
        return ans;
    }

}
