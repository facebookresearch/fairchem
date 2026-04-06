import {
  __esm
} from "/build/_shared/chunk-CGOEG7L2.js";

// ../../node_modules/@codemirror/legacy-modes/mode/forth.js
function toWordList(words) {
  var ret = [];
  words.split(" ").forEach(function(e) {
    ret.push({ name: e });
  });
  return ret;
}
function searchWordList(wordList, word) {
  var i;
  for (i = wordList.length - 1; i >= 0; i--) {
    if (wordList[i].name === word.toUpperCase()) {
      return wordList[i];
    }
  }
  return void 0;
}
var coreWordList, immediateWordList, forth;
var init_forth = __esm({
  "../../node_modules/@codemirror/legacy-modes/mode/forth.js"() {
    coreWordList = toWordList(
      "INVERT AND OR XOR 2* 2/ LSHIFT RSHIFT 0= = 0< < > U< MIN MAX 2DROP 2DUP 2OVER 2SWAP ?DUP DEPTH DROP DUP OVER ROT SWAP >R R> R@ + - 1+ 1- ABS NEGATE S>D * M* UM* FM/MOD SM/REM UM/MOD */ */MOD / /MOD MOD HERE , @ ! CELL+ CELLS C, C@ C! CHARS 2@ 2! ALIGN ALIGNED +! ALLOT CHAR [CHAR] [ ] BL FIND EXECUTE IMMEDIATE COUNT LITERAL STATE ; DOES> >BODY EVALUATE SOURCE >IN <# # #S #> HOLD SIGN BASE >NUMBER HEX DECIMAL FILL MOVE . CR EMIT SPACE SPACES TYPE U. .R U.R ACCEPT TRUE FALSE <> U> 0<> 0> NIP TUCK ROLL PICK 2>R 2R@ 2R> WITHIN UNUSED MARKER I J TO COMPILE, [COMPILE] SAVE-INPUT RESTORE-INPUT PAD ERASE 2LITERAL DNEGATE D- D+ D0< D0= D2* D2/ D< D= DMAX DMIN D>S DABS M+ M*/ D. D.R 2ROT DU< CATCH THROW FREE RESIZE ALLOCATE CS-PICK CS-ROLL GET-CURRENT SET-CURRENT FORTH-WORDLIST GET-ORDER SET-ORDER PREVIOUS SEARCH-WORDLIST WORDLIST FIND ALSO ONLY FORTH DEFINITIONS ORDER -TRAILING /STRING SEARCH COMPARE CMOVE CMOVE> BLANK SLITERAL"
    );
    immediateWordList = toWordList("IF ELSE THEN BEGIN WHILE REPEAT UNTIL RECURSE [IF] [ELSE] [THEN] ?DO DO LOOP +LOOP UNLOOP LEAVE EXIT AGAIN CASE OF ENDOF ENDCASE");
    forth = {
      name: "forth",
      startState: function() {
        return {
          state: "",
          base: 10,
          coreWordList,
          immediateWordList,
          wordList: []
        };
      },
      token: function(stream, stt) {
        var mat;
        if (stream.eatSpace()) {
          return null;
        }
        if (stt.state === "") {
          if (stream.match(/^(\]|:NONAME)(\s|$)/i)) {
            stt.state = " compilation";
            return "builtin";
          }
          mat = stream.match(/^(\:)\s+(\S+)(\s|$)+/);
          if (mat) {
            stt.wordList.push({ name: mat[2].toUpperCase() });
            stt.state = " compilation";
            return "def";
          }
          mat = stream.match(/^(VARIABLE|2VARIABLE|CONSTANT|2CONSTANT|CREATE|POSTPONE|VALUE|WORD)\s+(\S+)(\s|$)+/i);
          if (mat) {
            stt.wordList.push({ name: mat[2].toUpperCase() });
            return "def";
          }
          mat = stream.match(/^(\'|\[\'\])\s+(\S+)(\s|$)+/);
          if (mat) {
            return "builtin";
          }
        } else {
          if (stream.match(/^(\;|\[)(\s)/)) {
            stt.state = "";
            stream.backUp(1);
            return "builtin";
          }
          if (stream.match(/^(\;|\[)($)/)) {
            stt.state = "";
            return "builtin";
          }
          if (stream.match(/^(POSTPONE)\s+\S+(\s|$)+/)) {
            return "builtin";
          }
        }
        mat = stream.match(/^(\S+)(\s+|$)/);
        if (mat) {
          if (searchWordList(stt.wordList, mat[1]) !== void 0) {
            return "variable";
          }
          if (mat[1] === "\\") {
            stream.skipToEnd();
            return "comment";
          }
          if (searchWordList(stt.coreWordList, mat[1]) !== void 0) {
            return "builtin";
          }
          if (searchWordList(stt.immediateWordList, mat[1]) !== void 0) {
            return "keyword";
          }
          if (mat[1] === "(") {
            stream.eatWhile(function(s) {
              return s !== ")";
            });
            stream.eat(")");
            return "comment";
          }
          if (mat[1] === ".(") {
            stream.eatWhile(function(s) {
              return s !== ")";
            });
            stream.eat(")");
            return "string";
          }
          if (mat[1] === 'S"' || mat[1] === '."' || mat[1] === 'C"') {
            stream.eatWhile(function(s) {
              return s !== '"';
            });
            stream.eat('"');
            return "string";
          }
          if (mat[1] - 68719476735) {
            return "number";
          }
          return "atom";
        }
      }
    };
  }
});
init_forth();
export {
  forth
};
//# sourceMappingURL=/build/_shared/forth-PUDXV67F.js.map
