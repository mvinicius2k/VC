# Questão 2
import numpy as np
import cv2
from matplotlib import pyplot as plt

# branco é 0 e preto é 1
def pixel2word(px):
  if px == 0:
    return "1"
  else:
    return "0"


# Mapa dos binários e valores
dic = {

"11011001100"	:	[	"space"	,	"space"	,	"00"],
"11001101100"	:	[	"!","!","01"],
"11001100110"	:	[	"\"","\"","02"],
"10010011000"	:	[	"#","#","03"],
"10010001100"	:	[	"$","$","04"],
"10001001100"	:	[	"%","%","05"],
"10011001000"	:	[	"&","&","06"],
"10011000100"	:	[	"'","'","07"],
"10001100100"	:	[	"(","(","08"],
"11001001000"	:	[	")",")","09"],
"11001000100"	:	[	"*","*","10"],
"11000100100"	:	[	"+","+","11"],
"10110011100"	:	["",",",",	12"],
"10011011100"	:	[	"-","-","13"],
"10011001110"	:	[	".",".","14"],
"10111001100"	:	[	"/","/","15"],
"10011101100"	:	[	"0","0","16"],
"10011100110"	:	[	"1","1","17"],
"11001110010"	:	[	"2","2","18"],
"11001011100"	:	[	"3","3","19"],
"11001001110"	:	[	"4","4","20"],
"11011100100"	:	[	"5","5","21"],
"11001110100"	:	[	"6","6","22"],
"11101101110"	:	[	"7","7","23"],
"11101001100"	:	[	"8","8","24"],
"11100101100"	:	[	"9","9","25"],
"11100100110"	:	[	":",":","26"],
"11101100100"	:	[	";",";","27"],
"11100110100"	:	[	"<","<","28"],
"11100110010"	:	[	"=","=","29"],
"11011011000"	:	[	">",">","30"],
"11011000110"	:	[	"?","?","31"],
"11000110110"	:	[	"@","@","32"],
"10100011000"	:	[	"A","A","33"],
"10001011000"	:	[	"B","B","34"],
"10001000110"	:	[	"C","C","35"],
"10110001000"	:	[	"D","D","36"],
"10001101000"	:	[	"E","E","37"],
"10001100010"	:	[	"F","F","38"],
"11010001000"	:	[	"G","G","39"],
"11000101000"	:	[	"H","H","40"],
"11000100010"	:	[	"j","j","41"],
"10110111000"	:	[	"J","J","42"],
"10110001110"	:	[	"K","K","43"],
"10001101110"	:	[	"L","L","44"],
"10111011000"	:	[	"M","M","45"],
"10111000110"	:	[	"N","N","46"],
"10001110110"	:	[	"O","O","47"],
"11101110110"	:	[	"P","P","48"],
"11010001110"	:	[	"Q","Q","49"],
"11000101110"	:	[	"R","R","50"],
"11011101000"	:	[	"S","S","51"],
"11011100010"	:	[	"T","T","52"],
"11011101110"	:	[	"U","U","53"],
"11101011000"	:	[	"V","V","54"],
"11101000110"	:	[	"W","W","55"],
"11100010110"	:	[	"X","X","56"],
"11101101000"	:	[	"Y","Y","57"],
"11101100010"	:	[	"Z","Z","58"],
"11100011010"	:	[	"[","[","59"],
"11101111010"	:	[	"\\","\\","60"],
"11001000010"	:	[	"]","]","61"],
"11110001010"	:	[	"^","^","62"],
"10100110000"	:	[	"_","_","63"],
"10100001100"	:	[	"NUL","`","64"],
"10010110000"	:	[	"SOH","a","65"],
"10010000110"	:	[	"STX","b","66"],
"10000101100"	:	[	"ETX","c","67"],
"10000100110"	:	[	"EOT","d","68"],
"10110010000"	:	[	"ENQ","e","69"],
"10110000100"	:	[	"ACK","f","70"],
"10011010000"	:	[	"BEL","g","71"],
"10011000010"	:	[	"BS","h","72"],
"10000110100"	:	[	"HT","j","73"],
"10000110010"	:	[	"LF","j","74"],
"11000010010"	:	[	"VT","k","75"],
"11001010000"	:	[	"FF","l","76"],
"11110111010"	:	[	"CR","m","77"],
"11000010100"	:	[	"SO","n","78"],
"10001111010"	:	[	"SI","o","79"],
"10100111100"	:	[	"DLE","p","80"],
"10010111100"	:	[	"DC1","q","81"],
"10010011110"	:	[	"DC2","r","82"],
"10111100100"	:	[	"DC3","s","83"],
"10011110100"	:	[	"DC4","t","84"],
"10011110010"	:	[	"NAK","u","85"],
"11110100100"	:	[	"SYN","v","86"],
"11110010100"	:	[	"ETB","w","87"],
"11110010010"	:	[	"CAN","x","88"],
"11011011110"	:	[	"EM","y","89"],
"11011110110"	:	[	"SUB","z","90"],
"11110110110"	:	[	"ESC","{","91"],
"10101111000"	:	[	"FS","|","92"],
"10100011110"	:	[	"GS","}","93"],
"10001011110"	:	[	"RS","~","94"],
"10111101000"	:	[	"US","DEL","95"],
"10111100010"	:	[	"FNC 3","FNC 3","96"],
"11110101000"	:	[	"FNC 2","FNC 2","97"],
"11110100010"	:	[	"Shift B","Shift A","98"],
"10111011110"	:	[	"Code C","Code C","99"],
"10111101110"	:	[	"Code B","FNC 4","Code B"],
"11101011110"	:	[	"FNC 4","Code A","Code A"],
"11110101110"	:	[	"FNC 1","FNC 1","FNC 1"],
"11010000100"	:	[	"Start Code A"	,	"Start Code A"	,	"Start Code A"],
"11010010000"	:	[	"Start Code B"	,	"Start Code B"	,	"Start Code B"],
"11010011100"	:	[	"Start Code C"	,	"Start Code C"	,	"Start Code C"],
"11000111010"	:	[	"Stop"	,	"Stop" , "Stop"],
"11010111000"	:	[	"RStop"	,	"RStop"	,	"RStop"],
"1100011101011" :	[	"StopP"	,	"StopP" , "StopP"],

}

# "Enum" para cada estado de leitura
class State:

  SILENT = 1
  START = 2
  CODE_A = 3
  CODE_B = 4
  CODE_C = 5


state = State.SILENT

filename = "imagens/barcode-code-128.png"
img = cv2.imread(filename)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_, black_white = cv2.threshold(img, 200,255, cv2.THRESH_BINARY)


result = []
word = "" #string para listar os valores binários lidos
current_code = 0 # código a, b ou c


sum = 0 #somatório para checar início das checagens
weight = 1 #peso usado no sum
values = list(dic.values())
keys = list(dic.keys())



for i in range(0, img.shape[0]):
  j = 0
  while j < img.shape[1]:
    row = black_white[i]
    match state:
      case State.SILENT: #zona silent, pular todos os pixels até encontrar um preto
        if row[j] == 0:
          state = State.START
          j-= 1
        else:
          pass
      case State.START: #iniciando leitura
        word += pixel2word(row[j])
        j+= 1
      

    
    if word in keys: #se uma sequencia de binários coinscidiu com o que tá lá no mapa...

      if(keys[sum % 103] == word): #hora do checksum, finalizando
        result_str = "".join(map(str, result))
        print(result_str)
        exit()

      sum += keys.index(word) * weight #somando checksum


      match dic[word][current_code]: #verificando se é código para mudar a forma de leitura ou somente leitura simples
        case "Start Code A":
          current_code = 0
        case "Start Code B":
          current_code = 1
        case "Start Code C":
          current_code = 2
        case "Code A":
          current_code = 0
          weight += 1    
        case "Code B":
          current_code = 1
          weight += 1
        case "Code C":
          current_code = 2
          weight += 1
        case _:
          result.append(dic[word][current_code])
          weight += 1

      word = "" #palavra escrita na lista, limpando..
      
    j += 1

