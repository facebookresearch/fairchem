import {
  __esm
} from "/build/_shared/chunk-CGOEG7L2.js";

// ../../node_modules/@codemirror/legacy-modes/mode/pig.js
function words(str) {
  var obj = {}, words2 = str.split(" ");
  for (var i = 0; i < words2.length; ++i)
    obj[words2[i]] = true;
  return obj;
}
function chain(stream, state, f) {
  state.tokenize = f;
  return f(stream, state);
}
function tokenComment(stream, state) {
  var isEnd = false;
  var ch;
  while (ch = stream.next()) {
    if (ch == "/" && isEnd) {
      state.tokenize = tokenBase;
      break;
    }
    isEnd = ch == "*";
  }
  return "comment";
}
function tokenString(quote) {
  return function(stream, state) {
    var escaped = false, next, end = false;
    while ((next = stream.next()) != null) {
      if (next == quote && !escaped) {
        end = true;
        break;
      }
      escaped = !escaped && next == "\\";
    }
    if (end || !escaped)
      state.tokenize = tokenBase;
    return "error";
  };
}
function tokenBase(stream, state) {
  var ch = stream.next();
  if (ch == '"' || ch == "'")
    return chain(stream, state, tokenString(ch));
  else if (/[\[\]{}\(\),;\.]/.test(ch))
    return null;
  else if (/\d/.test(ch)) {
    stream.eatWhile(/[\w\.]/);
    return "number";
  } else if (ch == "/") {
    if (stream.eat("*")) {
      return chain(stream, state, tokenComment);
    } else {
      stream.eatWhile(isOperatorChar);
      return "operator";
    }
  } else if (ch == "-") {
    if (stream.eat("-")) {
      stream.skipToEnd();
      return "comment";
    } else {
      stream.eatWhile(isOperatorChar);
      return "operator";
    }
  } else if (isOperatorChar.test(ch)) {
    stream.eatWhile(isOperatorChar);
    return "operator";
  } else {
    stream.eatWhile(/[\w\$_]/);
    if (keywords && keywords.propertyIsEnumerable(stream.current().toUpperCase())) {
      if (!stream.eat(")") && !stream.eat("."))
        return "keyword";
    }
    if (builtins && builtins.propertyIsEnumerable(stream.current().toUpperCase()))
      return "builtin";
    if (types && types.propertyIsEnumerable(stream.current().toUpperCase()))
      return "type";
    return "variable";
  }
}
var pBuiltins, pKeywords, pTypes, builtins, keywords, types, isOperatorChar, pig;
var init_pig = __esm({
  "../../node_modules/@codemirror/legacy-modes/mode/pig.js"() {
    pBuiltins = "ABS ACOS ARITY ASIN ATAN AVG BAGSIZE BINSTORAGE BLOOM BUILDBLOOM CBRT CEIL CONCAT COR COS COSH COUNT COUNT_STAR COV CONSTANTSIZE CUBEDIMENSIONS DIFF DISTINCT DOUBLEABS DOUBLEAVG DOUBLEBASE DOUBLEMAX DOUBLEMIN DOUBLEROUND DOUBLESUM EXP FLOOR FLOATABS FLOATAVG FLOATMAX FLOATMIN FLOATROUND FLOATSUM GENERICINVOKER INDEXOF INTABS INTAVG INTMAX INTMIN INTSUM INVOKEFORDOUBLE INVOKEFORFLOAT INVOKEFORINT INVOKEFORLONG INVOKEFORSTRING INVOKER ISEMPTY JSONLOADER JSONMETADATA JSONSTORAGE LAST_INDEX_OF LCFIRST LOG LOG10 LOWER LONGABS LONGAVG LONGMAX LONGMIN LONGSUM MAX MIN MAPSIZE MONITOREDUDF NONDETERMINISTIC OUTPUTSCHEMA  PIGSTORAGE PIGSTREAMING RANDOM REGEX_EXTRACT REGEX_EXTRACT_ALL REPLACE ROUND SIN SINH SIZE SQRT STRSPLIT SUBSTRING SUM STRINGCONCAT STRINGMAX STRINGMIN STRINGSIZE TAN TANH TOBAG TOKENIZE TOMAP TOP TOTUPLE TRIM TEXTLOADER TUPLESIZE UCFIRST UPPER UTF8STORAGECONVERTER ";
    pKeywords = "VOID IMPORT RETURNS DEFINE LOAD FILTER FOREACH ORDER CUBE DISTINCT COGROUP JOIN CROSS UNION SPLIT INTO IF OTHERWISE ALL AS BY USING INNER OUTER ONSCHEMA PARALLEL PARTITION GROUP AND OR NOT GENERATE FLATTEN ASC DESC IS STREAM THROUGH STORE MAPREDUCE SHIP CACHE INPUT OUTPUT STDERROR STDIN STDOUT LIMIT SAMPLE LEFT RIGHT FULL EQ GT LT GTE LTE NEQ MATCHES TRUE FALSE DUMP";
    pTypes = "BOOLEAN INT LONG FLOAT DOUBLE CHARARRAY BYTEARRAY BAG TUPLE MAP ";
    builtins = words(pBuiltins);
    keywords = words(pKeywords);
    types = words(pTypes);
    isOperatorChar = /[*+\-%<>=&?:\/!|]/;
    pig = {
      name: "pig",
      startState: function() {
        return {
          tokenize: tokenBase,
          startOfLine: true
        };
      },
      token: function(stream, state) {
        if (stream.eatSpace())
          return null;
        var style = state.tokenize(stream, state);
        return style;
      },
      languageData: {
        autocomplete: (pBuiltins + pTypes + pKeywords).split(" ")
      }
    };
  }
});
init_pig();
export {
  pig
};
//# sourceMappingURL=/build/_shared/pig-3SNMDAPC.js.map
