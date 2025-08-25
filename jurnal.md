1. untuk membuat CNN model dengan block block, jangan gunakan koma diantara block block, hal ini akan menyebabkan model menganggap forward pass sebagai tuple
2. untuk forward pass yang digunakan sebaiknya data dummy,dataTrain,dataTest memiliki 4 dimensi contoh shape image: (1, 1, 28, 28)
hal ini disarankan,karena jika tidak maka color channel first pada shape index[1] akan dianggap model sebagai batch dan akan menghasilkan output pada layer sebelum classifier berbeda dengan yang di inginkan (batch, color_channel,height, widht) or (batch,height,widht,color_channel)
3. troubleshoot sebelum mengambil keputusan manipulasi tensor dengan melihat setiap shape(masalah umum dalam ML)
4. Pandas tidak menerima input dalam bentuk non numerical,hal ini mengharuskan kamu untuk mengubah inputnya terlebih dahulu
5. Num_worker menambah beban kinerja cpu namun memberikan kecepatan pemrosesan






!! cek note pada inline code yang tersebar