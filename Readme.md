```python
! pip3 install openpyxl pandas matplotlib seaborn numpy
```

    [33mDEPRECATION: Configuring installation scheme with distutils config files is deprecated and will no longer work in the near future. If you are using a Homebrew or Linuxbrew Python, please see discussion at https://github.com/Homebrew/homebrew-core/issues/76621[0m[33m
    [0mCollecting openpyxl
      Using cached openpyxl-3.1.2-py2.py3-none-any.whl (249 kB)
    Collecting et-xmlfile
      Using cached et_xmlfile-1.1.0-py3-none-any.whl (4.7 kB)
    Installing collected packages: et-xmlfile, openpyxl
    [33m  DEPRECATION: Configuring installation scheme with distutils config files is deprecated and will no longer work in the near future. If you are using a Homebrew or Linuxbrew Python, please see discussion at https://github.com/Homebrew/homebrew-core/issues/76621[0m[33m
    [0m[33m  DEPRECATION: Configuring installation scheme with distutils config files is deprecated and will no longer work in the near future. If you are using a Homebrew or Linuxbrew Python, please see discussion at https://github.com/Homebrew/homebrew-core/issues/76621[0m[33m
    [0m[33mDEPRECATION: Configuring installation scheme with distutils config files is deprecated and will no longer work in the near future. If you are using a Homebrew or Linuxbrew Python, please see discussion at https://github.com/Homebrew/homebrew-core/issues/76621[0m[33m
    [0mSuccessfully installed et-xmlfile-1.1.0 openpyxl-3.1.2
    
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m A new release of pip available: [0m[31;49m22.3.1[0m[39;49m -> [0m[32;49m23.0.1[0m
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m To update, run: [0m[32;49mpython3.9 -m pip install --upgrade pip[0m



```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import re
import pandas as pd
import seaborn as sns
sns.set_theme()
```


# OpenSSL Benchmarking Reports
### Bench Configurations
We have benchmarked the following Compute Optimized Azure VMs using **OpenSSL 1.0.2k-fips**:
 - **F16s_v2**: Intel(R) Xeon(R) Platinum 8370C CPU @ 2.80GHz
 - **F32s_v2**: Intel(R) Xeon(R) Platinum 8370C CPU @ 2.80GHz
 - **F48s_v2**: Intel(R) Xeon(R) Platinum 8272CL CPU @ 2.60GHz
 - **F64s_v2**: Intel(R) Xeon(R) Platinum 8370C CPU @ 2.80GHz
 
For detailed info on machine arch refer to report files or https://learn.microsoft.com/en-us/azure/virtual-machines/fsv2-series. 

## Code for parsing & plotting

### Initialization Values


```python
columns = [['Algorithm', '16B', '64B', '256B','1024B', '8192B'],
           ['Algorithm', 'Sign', 'Verify', 'Sign Rate', 'Verify Rate']]
regex_list = [r"^(\S+\s*\S+)\s+(\d+\.\d+)k\s+(\d+\.\d+)k\s+(\d+\.\d+)k\s+(\d+\.\d+)k\s+(\d+\.\d+)k.*$",
              r"^(.*)\s+(\d+\.\d+)s\s+(\d+\.\d+)s\s+(\d+\.\d+)\s+(\d+\.\d+).*$"]
vm = {
    'f48.txt':'F48sv2',
    'f64.txt':'F64sv2',
    'f16.txt':'F16sv2',
    'f32.txt':'F32sv2'
}
df_by_vm = {
    'F64sv2':{},
    'F48sv2':{},
    'F32sv2':{},
    'F16sv2':{}
}
```

### Dataframe by File Parsing 


```python
def get_df(filename, regex, columns, is_enc):
    output = open(filename, "r").read()
    matches = re.findall(regex, output, re.MULTILINE)
    df = pd.DataFrame(matches, columns=columns)
    if is_enc==0:
        df[columns[1:]] = (df[columns[1:]].astype(float)*1000/pow(2, 30))
    else:
        df[columns[1:]] = (df[columns[1:]].astype(float))
        df[columns[0]]= df[columns[0]].str.strip()
    df = df.dropna()[columns]
    df_by_vm[vm[filename]][is_enc]=df
    return df
```


```python
dfs=[]
for filename, vm_name in vm.items():
    for i in range(len(regex_list)):
        df = get_df(filename, regex_list[i], columns[i],i)
        dfs.append(df)
```


```python
dfs[0].head(50)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Algorithm</th>
      <th>16B</th>
      <th>64B</th>
      <th>256B</th>
      <th>1024B</th>
      <th>8192B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>md2</td>
      <td>0.003129</td>
      <td>0.006369</td>
      <td>0.008537</td>
      <td>0.009447</td>
      <td>0.009687</td>
    </tr>
    <tr>
      <th>1</th>
      <td>md4</td>
      <td>0.088672</td>
      <td>0.267040</td>
      <td>0.618644</td>
      <td>0.925659</td>
      <td>1.062711</td>
    </tr>
    <tr>
      <th>2</th>
      <td>md5</td>
      <td>0.064346</td>
      <td>0.185936</td>
      <td>0.400034</td>
      <td>0.562849</td>
      <td>0.648127</td>
    </tr>
    <tr>
      <th>3</th>
      <td>hmac(md5)</td>
      <td>0.052659</td>
      <td>0.160094</td>
      <td>0.366562</td>
      <td>0.548038</td>
      <td>0.630613</td>
    </tr>
    <tr>
      <th>4</th>
      <td>sha1</td>
      <td>0.075936</td>
      <td>0.220907</td>
      <td>0.510344</td>
      <td>0.770711</td>
      <td>0.900429</td>
    </tr>
    <tr>
      <th>5</th>
      <td>rmd160</td>
      <td>0.042133</td>
      <td>0.101315</td>
      <td>0.182006</td>
      <td>0.230427</td>
      <td>0.242778</td>
    </tr>
    <tr>
      <th>6</th>
      <td>rc4</td>
      <td>0.619633</td>
      <td>0.650601</td>
      <td>0.545359</td>
      <td>0.517333</td>
      <td>0.498189</td>
    </tr>
    <tr>
      <th>7</th>
      <td>des cbc</td>
      <td>0.071575</td>
      <td>0.073745</td>
      <td>0.074051</td>
      <td>0.074350</td>
      <td>0.073329</td>
    </tr>
    <tr>
      <th>8</th>
      <td>des ede3</td>
      <td>0.027640</td>
      <td>0.027557</td>
      <td>0.028330</td>
      <td>0.028542</td>
      <td>0.028069</td>
    </tr>
    <tr>
      <th>9</th>
      <td>idea cbc</td>
      <td>0.086304</td>
      <td>0.089621</td>
      <td>0.090901</td>
      <td>0.090699</td>
      <td>0.091456</td>
    </tr>
    <tr>
      <th>10</th>
      <td>seed cbc</td>
      <td>0.075369</td>
      <td>0.076608</td>
      <td>0.076585</td>
      <td>0.076884</td>
      <td>0.075615</td>
    </tr>
    <tr>
      <th>11</th>
      <td>rc2 cbc</td>
      <td>0.044727</td>
      <td>0.045554</td>
      <td>0.045703</td>
      <td>0.045978</td>
      <td>0.045998</td>
    </tr>
    <tr>
      <th>12</th>
      <td>rc5-32/12 cbc</td>
      <td>0.198362</td>
      <td>0.220558</td>
      <td>0.226453</td>
      <td>0.228168</td>
      <td>0.222577</td>
    </tr>
    <tr>
      <th>13</th>
      <td>blowfish cbc</td>
      <td>0.122483</td>
      <td>0.131394</td>
      <td>0.132911</td>
      <td>0.133846</td>
      <td>0.134514</td>
    </tr>
    <tr>
      <th>14</th>
      <td>cast cbc</td>
      <td>0.114204</td>
      <td>0.122192</td>
      <td>0.123099</td>
      <td>0.124329</td>
      <td>0.122231</td>
    </tr>
    <tr>
      <th>15</th>
      <td>aes-128 cbc</td>
      <td>0.129304</td>
      <td>0.145100</td>
      <td>0.146992</td>
      <td>0.149972</td>
      <td>0.150711</td>
    </tr>
    <tr>
      <th>16</th>
      <td>aes-192 cbc</td>
      <td>0.110656</td>
      <td>0.120807</td>
      <td>0.123398</td>
      <td>0.123587</td>
      <td>0.121709</td>
    </tr>
    <tr>
      <th>17</th>
      <td>aes-256 cbc</td>
      <td>0.095120</td>
      <td>0.102914</td>
      <td>0.105400</td>
      <td>0.106317</td>
      <td>0.106885</td>
    </tr>
    <tr>
      <th>18</th>
      <td>camellia-128 cbc</td>
      <td>0.107843</td>
      <td>0.166033</td>
      <td>0.187146</td>
      <td>0.193413</td>
      <td>0.192187</td>
    </tr>
    <tr>
      <th>19</th>
      <td>camellia-192 cbc</td>
      <td>0.094209</td>
      <td>0.127578</td>
      <td>0.139131</td>
      <td>0.142531</td>
      <td>0.145345</td>
    </tr>
    <tr>
      <th>20</th>
      <td>camellia-256 cbc</td>
      <td>0.093984</td>
      <td>0.127510</td>
      <td>0.139048</td>
      <td>0.143518</td>
      <td>0.141586</td>
    </tr>
    <tr>
      <th>21</th>
      <td>sha256</td>
      <td>0.077052</td>
      <td>0.172359</td>
      <td>0.315463</td>
      <td>0.394272</td>
      <td>0.412453</td>
    </tr>
    <tr>
      <th>22</th>
      <td>sha512</td>
      <td>0.053687</td>
      <td>0.217880</td>
      <td>0.365012</td>
      <td>0.539430</td>
      <td>0.625478</td>
    </tr>
    <tr>
      <th>23</th>
      <td>whirlpool</td>
      <td>0.036049</td>
      <td>0.077681</td>
      <td>0.128911</td>
      <td>0.154386</td>
      <td>0.161263</td>
    </tr>
    <tr>
      <th>24</th>
      <td>aes-128 ige</td>
      <td>0.132081</td>
      <td>0.138685</td>
      <td>0.139170</td>
      <td>0.140550</td>
      <td>0.137327</td>
    </tr>
    <tr>
      <th>25</th>
      <td>aes-192 ige</td>
      <td>0.112562</td>
      <td>0.116530</td>
      <td>0.117198</td>
      <td>0.118085</td>
      <td>0.118179</td>
    </tr>
    <tr>
      <th>26</th>
      <td>aes-256 ige</td>
      <td>0.097677</td>
      <td>0.100968</td>
      <td>0.101140</td>
      <td>0.101350</td>
      <td>0.099045</td>
    </tr>
    <tr>
      <th>27</th>
      <td>ghash</td>
      <td>1.317272</td>
      <td>5.230101</td>
      <td>7.979045</td>
      <td>8.756001</td>
      <td>9.040395</td>
    </tr>
    <tr>
      <th>28</th>
      <td>aes-128-cbc</td>
      <td>1.089691</td>
      <td>1.199371</td>
      <td>1.190869</td>
      <td>1.222647</td>
      <td>1.229479</td>
    </tr>
    <tr>
      <th>29</th>
      <td>aes-192-cbc</td>
      <td>0.939595</td>
      <td>1.001947</td>
      <td>1.018783</td>
      <td>1.016890</td>
      <td>1.037120</td>
    </tr>
    <tr>
      <th>30</th>
      <td>aes-256-cbc</td>
      <td>0.808393</td>
      <td>0.871557</td>
      <td>0.866819</td>
      <td>0.882868</td>
      <td>0.898811</td>
    </tr>
    <tr>
      <th>31</th>
      <td>aes-128-gcm</td>
      <td>0.619298</td>
      <td>1.422761</td>
      <td>2.678002</td>
      <td>4.048132</td>
      <td>5.019335</td>
    </tr>
    <tr>
      <th>32</th>
      <td>aes-192-gcm</td>
      <td>0.560195</td>
      <td>1.266597</td>
      <td>2.306938</td>
      <td>3.515232</td>
      <td>4.205828</td>
    </tr>
    <tr>
      <th>33</th>
      <td>aes-256-gcm</td>
      <td>0.555007</td>
      <td>1.330763</td>
      <td>2.219283</td>
      <td>3.139349</td>
      <td>3.620478</td>
    </tr>
  </tbody>
</table>
</div>




```python
enc=0
df_aes128 = dfs[enc].loc[dfs[enc]['Algorithm'].isin(['aes-128-gcm','aes-128-cbc','aes-128 cbc'])]
df_aes128.head()

df_aes192 = dfs[enc].loc[dfs[enc]['Algorithm'].isin(['aes-192-gcm','aes-192-cbc','aes-192 cbc'])]
df_aes192.head()

df_aes256 = dfs[enc].loc[dfs[enc]['Algorithm'].isin(['aes-256-gcm','aes-256-cbc','aes-256 cbc'])]
df_aes256.head()

df_md = dfs[enc].loc[dfs[enc]['Algorithm'].isin(['md5','sha1','sha256', 'sha512'])]
df_md.head()

aes_list = {128:df_aes128, 192:df_aes192, 256:df_aes256} 
```


```python
sign=1
df_rsa = dfs[sign].loc[dfs[sign]['Algorithm'].isin(['rsa  512 bits','rsa 1024 bits','rsa 2048 bits','rsa 4096 bits'])]
df_rsa.head()

df_ecdsa = dfs[sign].loc[dfs[sign]['Algorithm'].isin(['256 bit ecdsa (nistp256)','384 bit ecdsa (nistp384)','521 bit ecdsa (nistp521)'])]
df_ecdsa.head()

df_dsa = dfs[sign].loc[dfs[sign]['Algorithm'].isin(['dsa  512 bits','dsa 1024 bits','dsa 2048 bits'])]
df_dsa.head()

sign_list = {'RSA':df_rsa, 'ECDSA': df_ecdsa, 'DSA':df_dsa} 
```

### Plot functions

#### AES & MD Plot functions


```python
def get_plottable_df(df, enc):
    x=dict([(df[columns[enc][0]].iloc[i], list(df[columns[enc][1:]].iloc[i])) for i in range(len(df))])
    return x
```


```python
sns.set_style("whitegrid")
sns.set_context("paper")
block_size =  ['16B', '64B', '256B', '1024B', '8192B']
def plot_graph(df, enc, x_labels, title):
    aes = get_plottable_df(df, enc)
    x = np.arange(len(x_labels))  # the label locations
    width = 0.2  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in aes.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=0.1, fmt='%.1f')
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('GB per second')
    ax.set_title(title)
    ax.set_xticks(x + width, block_size)
    ax.legend(loc='upper left', ncols=3)
    # ax.set_ylim(0, 10)

    
def plot_all_enc_md(filename):
    for keysize, aes_df in aes_list.items():
        plot_graph(aes_df, enc, block_size, f'{vm[filename]}: AES-{keysize} processing speed by Block Size')

    plot_graph(df_md, enc, block_size, f'{vm[filename]}: Message Digest processing speed by Block Size')
    plt.show()
```

#### Signature Algorithm Plot functions


```python
# fig, ax = plt.subplots(layout='constrained')
# plt.bar(df['Algorithm'].to_list(),df['Verify Rate', 'Sign Rate'].to_list())
def plot_sign_graph(df, prefix):
    # Data
    algorithm = df['Algorithm'].to_list()
    sign_rate = df['Sign Rate'].to_list()
    verify_rate = df['Verify Rate'].to_list()

    # Create an array of indexes for each algorithm
    x_indexes = np.arange(len(algorithm))

    # Set the width of the bars
    bar_width = 0.35

    # Create the figure and axis objects
    fig, ax = plt.subplots(layout='constrained')

    # Plot the bars for sign rate and verify rate
    s=ax.bar(x_indexes - bar_width/2, sign_rate, bar_width, label='Sign Rate')
    v=ax.bar(x_indexes + bar_width/2, verify_rate, bar_width, label='Verify Rate')
    ax.bar_label(s, padding=0.2,fmt='%.f')
    ax.bar_label(v, padding=0.2,fmt='%.f')
    # Set the x-ticks and labels
    ax.set_xticks(x_indexes)
    ax.set_xticklabels(algorithm)
    ax.set_title(f'{prefix} Operations per second by Key Size')
    # Set the y-axis label
    ax.set_ylabel('Rate per second')

    
    # Add the legend
    ax.legend(ncols=2)

def plot_all_sign(filename):
    for algo, df_sign in sign_list.items():
        plot_sign_graph(df_sign, f'{vm[filename]}: {algo}')
    # Display the plot
    plt.show()

```


```python
def plot_all(filename):
    plot_all_enc_md(filename)
    plot_all_sign(filename)
```

## Graph Plots & Reports

### Azure F16sv2: Encryption & Message Digest Graphical Reports


```python
plot_all('f16.txt')
```


    
![png](images/output_21_0.png)
    



    
![png](images/output_21_1.png)
    



    
![png](images/output_21_2.png)
    



    
![png](images/output_21_3.png)
    



    
![png](images/output_21_4.png)
    



    
![png](images/output_21_5.png)
    



    
![png](images/output_21_6.png)
    


### Azure F32sv2: Encryption & Message Digest Graphical Reports


```python
plot_all('f32.txt')
```


    
![png](images/output_23_0.png)
    



    
![png](images/output_23_1.png)
    



    
![png](images/output_23_2.png)
    



    
![png](images/output_23_3.png)
    



    
![png](images/output_23_4.png)
    



    
![png](images/output_23_5.png)
    



    
![png](images/output_23_6.png)
    


### Azure F48sv2: Encryption & Message Digest Graphical Reports


```python
plot_all('f48.txt')
```


    
![png](images/output_25_0.png)
    



    
![png](images/output_25_1.png)
    



    
![png](images/output_25_2.png)
    



    
![png](images/output_25_3.png)
    



    
![png](images/output_25_4.png)
    



    
![png](images/output_25_5.png)
    



    
![png](images/output_25_6.png)
    


### Azure F64sv2: Encryption & Message Digest Graphical Reports


```python
plot_all('f64.txt')
```


    
![png](images/output_27_0.png)
    



    
![png](images/output_27_1.png)
    



    
![png](images/output_27_2.png)
    



    
![png](images/output_27_3.png)
    



    
![png](images/output_27_4.png)
    



    
![png](images/output_27_5.png)
    



    
![png](images/output_27_6.png)
    


### Overview of OpenSSL EVP Interface, AES-NI, SHA-NI
Openssl's EVP API provides a more streamlined interface for working with cryptographic functions, reducing the risk of errors and security vulnerabilities. For example, the EVP API automatically handles padding and block size issues. Additionally, the EVP API has the ability to take advantage of optimized code paths, CPU-specific optimizations and it can be configured to use hardware acceleration, such as the AES-NI instruction set on Intel CPUs, which can further improve performance. However, the exact extent to which the EVP API uses multiple cores will depend on a variety of factors, including the specific cryptographic function being used, the size and complexity of the input data, the hardware and software environment, and other factors. Therefore, it's important to test and benchmark the performance of OpenSSL's ciphers to determine whether it is taking advantage of multiple cores effectively.

Some CPUs have specialized instructions for performing SHA512 hashing, such as the SHA extensions or SHA-NI on Intel CPUs. If the CPU supports these instructions, SHA512 may be faster than SHA256 even for smaller data sets due to hardware acceleration. We found that these VMs don't have SHA-NI but have other relevant extensions like ssse3, avx, avx2, avx512 instruction set as visible in the output of `lscpu` command in the report files. 

### Conclusion on performance of AES-CBC vs AES-CBC(EVP) vs AES-GCM(EVP) 
 - AES-CBC, AES-CBC(EVP) & AES-GCM(EVP) is shown as 'aes-128 cbc', 'aes-128-cbc' & 'aes-128-gcm' respectively for all the graphs.
 - We see that AES-CBC has the lowest performance among the three. Also the performance remains the same across different block sizes.
 - We observe that AES-CBC throught EVP interface performs almost 10x better. This is expected as EVP can exploit optimized code paths & hardware acceleration using AES-NI.
 - AES-GCM is the preferred cipher as per the Adobe standard as it has added security advantages over CBC like authentication tag. We observe it also performs the best across all machines under consideration.  
 - Performance AES-GCM & the performance gap of AES-GCM & CBC(EVP) drastically increase as the block size increases.
 
### Conclusion on performance of HMAC(MD5) vs SHA1 vs SHA256 vs SHA512 
 - SHA1 is the best performing message digest function across all block sizes, we avoid this as the Adobe standard only allows SHA1 for HMAC for certain TLS ciphers & prefers SHA2 family of functions.
 - We observe that SHA512 is faster than SHA256 for larger block sizes. SHA256 produces a 256-bit hash, while SHA512 produces a 512-bit hash. This means that SHA512 can handle larger data sets more efficiently, as it requires fewer iterations to hash the same amount of data.
 - 

### Performance Across Azure VM Type


```python
df_sign=pd.DataFrame()
df_enc=pd.DataFrame()
for vm_name, algo in df_by_vm.items(): 
    df_by_vm[vm_name][sign]['vm']=vm_name    
    df_by_vm[vm_name][enc]['vm']=vm_name
    df_sign = pd.concat([df_sign,df_by_vm[vm_name][sign]])
    df_enc = pd.concat([df_enc,df_by_vm[vm_name][enc]])

y_labels = ['F64','F48','F32','F16']
```

#### AES-GCM Speed by VM Type


```python
x_labels = ['16B','64B','256B','1024B','8192B']
df_aesgcm = df_enc.loc[df_enc['Algorithm'].isin(['aes-128-gcm'])]
ax = sns.heatmap(np.array(df_aesgcm[x_labels].head()),xticklabels=x_labels, yticklabels=y_labels,annot=True, fmt='.2f')
ax.set_title('AES-GCM Speed in GBps')
ax.set_xlabel('Block Size')
ax.set_ylabel('Azure VMs')
```




    Text(52.91666666666667, 0.5, 'Azure VMs')




    
![png](images/output_32_1.png)
    


#### SHA256 Speed by VM Type


```python
x_labels = ['16B','64B','256B','1024B','8192B']
df_sha = df_enc.loc[df_enc['Algorithm'].isin(['sha256'])]
ax = sns.heatmap(np.array(df_sha[x_labels].head()),xticklabels=x_labels, yticklabels=y_labels,annot=True, fmt='.2f')
ax.set_title('SHA256 Speed in GBps')
ax.set_xlabel('Block Size')
ax.set_ylabel('Azure VMs')
```




    Text(52.91666666666667, 0.5, 'Azure VMs')




    
![png](images/output_34_1.png)
    


#### ECDSA Speed by VM Type


```python
df_ecdsa = df_sign.loc[df_sign['Algorithm'].isin(['256 bit ecdsa (nistp256)'])]
x_labels = ['Sign Rate','Verify Rate']
ax = sns.heatmap(np.array(df_ecdsa[x_labels].head()),xticklabels=x_labels, yticklabels=y_labels,annot=True, fmt='.2f')
ax.set_title('ECDSA Speed in operations/second')
ax.set_xlabel('Operation')
ax.set_ylabel('Azure VMs')
```




    Text(52.91666666666667, 0.5, 'Azure VMs')




    
![png](images/output_36_1.png)
    


### Conclusion on performance across Azure VM Types
 - From the heatmaps, we observe that the throughput is similar across all VM types for encryption, message digest and signature algorithms. Since we didn't use the "-multi" command line argument while benchmarking which uses multiple threads, this was expected. We might see considerable differences if we run OpenSSL with multiple threads on all the VMs.  
 - For F48sv2 VM AES-GCM is a little slower than the other VMs & ECDSA is faster than the other VMs. This could be due to the difference in hardware as F48sv2 machine had "Intel(R) Xeon(R) Platinum 8272CL CPU @ 2.60GHz" whereas other VMs had Intel(R) Xeon(R) Platinum 8370C CPU @ 2.80GHz.

### Performance & Key Size

#### AES-GCM Speed by Key Size


```python
df_aes_gcm = df_by_vm['F48sv2'][enc].loc[df_by_vm['F48sv2'][enc]['Algorithm'].isin(['aes-128-gcm','aes-192-gcm','aes-256-gcm'])]
df_aes_gcm[['16B','64B','256B','1024B','8192B']].head()
y_labels = [128,192,256]
x_labels = ['16B','64B','256B','1024B','8192B']
ax = sns.heatmap(np.array(df_aes_gcm[x_labels].head()),xticklabels=x_labels, yticklabels=y_labels,annot=True, fmt='.2f')
ax.set_title('F48sv2 AES-GCM Speed in GBps')
ax.set_xlabel('Block Size')
ax.set_ylabel('Key Size')
```




    Text(52.91666666666667, 0.5, 'Key Size')




    
![png](images/output_40_1.png)
    


#### AES CBC Speed by Key Size


```python
df_aes_cbc = df_by_vm['F48sv2'][enc].loc[df_by_vm['F48sv2'][enc]['Algorithm'].isin(['aes-128 cbc','aes-192 cbc','aes-256 cbc'])]
df_aes_cbc[['16B','64B','256B','1024B','8192B']].head()
y_labels = [128,192,256]
x_labels = ['16B','64B','256B','1024B','8192B']
ax = sns.heatmap(np.array(df_aes_cbc[x_labels].head()),xticklabels=x_labels, yticklabels=y_labels,annot=True, fmt='.2f')
ax.set_title('F48sv2 AES-CBC Speed in GBps')
ax.set_xlabel('Block Size')
ax.set_ylabel('Key Size')
```




    Text(52.91666666666667, 0.5, 'Key Size')




    
![png](images/output_42_1.png)
    


#### EVP AES-CBC Speed by Key Size


```python
df_aes_cbc_evp = df_by_vm['F48sv2'][enc].loc[df_by_vm['F48sv2'][enc]['Algorithm'].isin(['aes-128-cbc','aes-192-cbc','aes-256-cbc'])]
df_aes_cbc_evp[['16B','64B','256B','1024B','8192B']].head()
y_labels = [128,192,256]
x_labels = ['16B','64B','256B','1024B','8192B']
ax = sns.heatmap(np.array(df_aes_cbc_evp[x_labels].head()),xticklabels=x_labels, yticklabels=y_labels,annot=True, fmt='.2f')
ax.set_title('F48sv2 EVP AES-CBC Speed in GBps')
ax.set_xlabel('Block Size')
ax.set_ylabel('Key Size')
```




    Text(52.91666666666667, 0.5, 'Key Size')




    
![png](images/output_44_1.png)
    


#### ECDSA Speed by Key Size


```python
df_ecdsa = df_by_vm['F48sv2'][sign].loc[df_by_vm['F48sv2'][sign]['Algorithm'].isin(['256 bit ecdsa (nistp256)','384 bit ecdsa (nistp384)','521 bit ecdsa (nistp521)'])]
df_ecdsa.head()
y_labels = [256,384,521]
x_labels = ['Sign Rate','Verify Rate']
ax = sns.heatmap(np.array(df_ecdsa[x_labels].head()),xticklabels=x_labels, yticklabels=y_labels,annot=True, fmt='.2f')
ax.set_title('F48sv2 ECDSA Speed in operations/second')
ax.set_xlabel('Operation')
ax.set_ylabel('Key Size')
```




    Text(52.91666666666667, 0.5, 'Key Size')




    
![png](images/output_46_1.png)
    


#### RSA Speed by Key Size


```python
df_rsa = df_by_vm['F48sv2'][sign].loc[df_by_vm['F48sv2'][sign]['Algorithm'].isin(['rsa  512 bits','rsa 1024 bits','rsa 2048 bits','rsa 4096 bits'])]
df_rsa[['Sign Rate','Verify Rate']].head()
y_labels = [512,1024,2048,4096]
x_labels = ['Sign Rate','Verify Rate']
ax = sns.heatmap(np.array(df_rsa[x_labels].head()),xticklabels=x_labels, yticklabels=y_labels,annot=True, fmt='.2f')
ax.set_title('F48sv2 RSA Speed in operations/second')
ax.set_xlabel('Operation')
ax.set_ylabel('Key Size')
```




    Text(52.91666666666667, 0.5, 'Key Size')




    
![png](images/output_48_1.png)
    


#### DSA Speed by Key Size


```python
df_dsa = df_by_vm['F48sv2'][sign].loc[df_by_vm['F48sv2'][sign]['Algorithm'].isin(['dsa  512 bits','dsa 1024 bits','dsa 2048 bits'])]
df_dsa[['Sign Rate','Verify Rate']].head()
y_labels = [512,1024,2048]
x_labels = ['Sign Rate','Verify Rate']
ax = sns.heatmap(np.array(df_dsa[x_labels].head()),xticklabels=x_labels, yticklabels=y_labels,annot=True, fmt='.2f')
ax.set_title('F48sv2 DSA Speed in operations/second')
ax.set_xlabel('Operation')
ax.set_ylabel('Key Size')
```




    Text(52.91666666666667, 0.5, 'Key Size')




    
![png](images/images/output_50_1.png)
    


### Conclusions On Choice of Signature Algorithms
 - RSA & DSA are faster in verify operations than signing.
 - ECDSA is faster in sign operations than verification.
 - We know that bigger the key size, more the security. From the heatmaps of ECDSA & RSA we observe that with bigger key sizes throughput decreases. 
 
#### RSA vs ECDSA vs DSA
 - As per Internal Adobe standards we can only use RSA/DSA with 2048 bits key and higher or ECDSA with 256 bits key or higher. 
 - With this restriction, RSA-2048 is worse in performance than ECDSA-256 in signing i.e. 1.7K/s vs 30K/s. 
 - ECDSA-256 is better than DSA-2048 in both signing and verification.
 - Since the primary task of a server in a TLS session is signing the certificates the server should be optimized for signing. However, both signing and verifying operations can become computationally expensive, especially in high-volume environments like ours where MTAs receive 60 Million RPS of TLS email traffic from internal upstream services and need to send TLS email traffic to ISPs. In such cases, server hardware and software should be optimized for both signing and verifying operations to ensure efficient and reliable TLS connections. Since ECDSA performs well in signing & verification, we choose to prefer ECDSA-256 over RSA/DSA-2048 for server processes.


```python

```
