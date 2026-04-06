import {
  __esm
} from "/build/_shared/chunk-CGOEG7L2.js";

// ../../node_modules/@codemirror/legacy-modes/mode/oz.js
function wordRegexp(words) {
  return new RegExp("^((" + words.join(")|(") + "))\\b");
}
function tokenBase(stream, state) {
  if (stream.eatSpace()) {
    return null;
  }
  if (stream.match(/[{}]/)) {
    return "bracket";
  }
  if (stream.match("[]")) {
    return "keyword";
  }
  if (stream.match(tripleOperators) || stream.match(doubleOperators)) {
    return "operator";
  }
  if (stream.match(atoms)) {
    return "atom";
  }
  var matched = stream.match(openingKeywords);
  if (matched) {
    if (!state.doInCurrentLine)
      state.currentIndent++;
    else
      state.doInCurrentLine = false;
    if (matched[0] == "proc" || matched[0] == "fun")
      state.tokenize = tokenFunProc;
    else if (matched[0] == "class")
      state.tokenize = tokenClass;
    else if (matched[0] == "meth")
      state.tokenize = tokenMeth;
    return "keyword";
  }
  if (stream.match(middleKeywords) || stream.match(commonKeywords)) {
    return "keyword";
  }
  if (stream.match(endKeywords)) {
    state.currentIndent--;
    return "keyword";
  }
  var ch = stream.next();
  if (ch == '"' || ch == "'") {
    state.tokenize = tokenString(ch);
    return state.tokenize(stream, state);
  }
  if (/[~\d]/.test(ch)) {
    if (ch == "~") {
      if (!/^[0-9]/.test(stream.peek()))
        return null;
      else if (stream.next() == "0" && stream.match(/^[xX][0-9a-fA-F]+/) || stream.match(/^[0-9]*(\.[0-9]+)?([eE][~+]?[0-9]+)?/))
        return "number";
    }
    if (ch == "0" && stream.match(/^[xX][0-9a-fA-F]+/) || stream.match(/^[0-9]*(\.[0-9]+)?([eE][~+]?[0-9]+)?/))
      return "number";
    return null;
  }
  if (ch == "%") {
    stream.skipToEnd();
    return "comment";
  } else if (ch == "/") {
    if (stream.eat("*")) {
      state.tokenize = tokenComment;
      return tokenComment(stream, state);
    }
  }
  if (singleOperators.test(ch)) {
    return "operator";
  }
  stream.eatWhile(/\w/);
  return "variable";
}
function tokenClass(stream, state) {
  if (stream.eatSpace()) {
    return null;
  }
  stream.match(/([A-Z][A-Za-z0-9_]*)|(`.+`)/);
  state.tokenize = tokenBase;
  return "type";
}
function tokenMeth(stream, state) {
  if (stream.eatSpace()) {
    return null;
  }
  stream.match(/([a-zA-Z][A-Za-z0-9_]*)|(`.+`)/);
  state.tokenize = tokenBase;
  return "def";
}
function tokenFunProc(stream, state) {
  if (stream.eatSpace()) {
    return null;
  }
  if (!state.hasPassedFirstStage && stream.eat("{")) {
    state.hasPassedFirstStage = true;
    return "bracket";
  } else if (state.hasPassedFirstStage) {
    stream.match(/([A-Z][A-Za-z0-9_]*)|(`.+`)|\$/);
    state.hasPassedFirstStage = false;
    state.tokenize = tokenBase;
    return "def";
  } else {
    state.tokenize = tokenBase;
    return null;
  }
}
function tokenComment(stream, state) {
  var maybeEnd = false, ch;
  while (ch = stream.next()) {
    if (ch == "/" && maybeEnd) {
      state.tokenize = tokenBase;
      break;
    }
    maybeEnd = ch == "*";
  }
  return "comment";
}
function tokenString(quote) {
  return function(stream, state) {
    var escaped = false, next, end2 = false;
    while ((next = stream.next()) != null) {
      if (next == quote && !escaped) {
        end2 = true;
        break;
      }
      escaped = !escaped && next == "\\";
    }
    if (end2 || !escaped)
      state.tokenize = tokenBase;
    return "string";
  };
}
function buildElectricInputRegEx() {
  var allClosings = middle.concat(end);
  return new RegExp("[\\[\\]]|(" + allClosings.join("|") + ")$");
}
var singleOperators, doubleOperators, tripleOperators, middle, end, atoms, commonKeywords, openingKeywords, middleKeywords, endKeywords, oz;
var init_oz = __esm({
  "../../node_modules/@codemirror/legacy-modes/mode/oz.js"() {
    singleOperators = /[\^@!\|<>#~\.\*\-\+\\/,=]/;
    doubleOperators = /(<-)|(:=)|(=<)|(>=)|(<=)|(<:)|(>:)|(=:)|(\\=)|(\\=:)|(!!)|(==)|(::)/;
    tripleOperators = /(:::)|(\.\.\.)|(=<:)|(>=:)/;
    middle = [
      "in",
      "then",
      "else",
      "of",
      "elseof",
      "elsecase",
      "elseif",
      "catch",
      "finally",
      "with",
      "require",
      "prepare",
      "import",
      "export",
      "define",
      "do"
    ];
    end = ["end"];
    atoms = wordRegexp(["true", "false", "nil", "unit"]);
    commonKeywords = wordRegexp([
      "andthen",
      "at",
      "attr",
      "declare",
      "feat",
      "from",
      "lex",
      "mod",
      "div",
      "mode",
      "orelse",
      "parser",
      "prod",
      "prop",
      "scanner",
      "self",
      "syn",
      "token"
    ]);
    openingKeywords = wordRegexp([
      "local",
      "proc",
      "fun",
      "case",
      "class",
      "if",
      "cond",
      "or",
      "dis",
      "choice",
      "not",
      "thread",
      "try",
      "raise",
      "lock",
      "for",
      "suchthat",
      "meth",
      "functor"
    ]);
    middleKeywords = wordRegexp(middle);
    endKeywords = wordRegexp(end);
    oz = {
      name: "oz",
      startState: function() {
        return {
          tokenize: tokenBase,
          currentIndent: 0,
          doInCurrentLine: false,
          hasPassedFirstStage: false
        };
      },
      token: function(stream, state) {
        if (stream.sol())
          state.doInCurrentLine = 0;
        return state.tokenize(stream, state);
      },
      indent: function(state, textAfter, cx) {
        var trueText = textAfter.replace(/^\s+|\s+$/g, "");
        if (trueText.match(endKeywords) || trueText.match(middleKeywords) || trueText.match(/(\[])/))
          return cx.unit * (state.currentIndent - 1);
        if (state.currentIndent < 0)
          return 0;
        return state.currentIndent * cx.unit;
      },
      languageData: {
        indentOnInut: buildElectricInputRegEx(),
        commentTokens: { line: "%", block: { open: "/*", close: "*/" } }
      }
    };
  }
});
init_oz();
export {
  oz
};
//# sourceMappingURL=/build/_shared/oz-JM6ISGKV.js.map
