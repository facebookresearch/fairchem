import {
  __esm
} from "/build/_shared/chunk-CGOEG7L2.js";

// ../../node_modules/@codemirror/legacy-modes/mode/scheme.js
function makeKeywords(str) {
  var obj = {}, words = str.split(" ");
  for (var i = 0; i < words.length; ++i)
    obj[words[i]] = true;
  return obj;
}
function stateStack(indent, type, prev) {
  this.indent = indent;
  this.type = type;
  this.prev = prev;
}
function pushStack(state, indent, type) {
  state.indentStack = new stateStack(indent, type, state.indentStack);
}
function popStack(state) {
  state.indentStack = state.indentStack.prev;
}
function isBinaryNumber(stream) {
  return stream.match(binaryMatcher);
}
function isOctalNumber(stream) {
  return stream.match(octalMatcher);
}
function isDecimalNumber(stream, backup) {
  if (backup === true) {
    stream.backUp(1);
  }
  return stream.match(decimalMatcher);
}
function isHexNumber(stream) {
  return stream.match(hexMatcher);
}
function processEscapedSequence(stream, options) {
  var next, escaped = false;
  while ((next = stream.next()) != null) {
    if (next == options.token && !escaped) {
      options.state.mode = false;
      break;
    }
    escaped = !escaped && next == "\\";
  }
}
var BUILTIN, COMMENT, STRING, SYMBOL, ATOM, NUMBER, BRACKET, INDENT_WORD_SKIP, keywords, indentKeys, binaryMatcher, octalMatcher, hexMatcher, decimalMatcher, scheme;
var init_scheme = __esm({
  "../../node_modules/@codemirror/legacy-modes/mode/scheme.js"() {
    BUILTIN = "builtin";
    COMMENT = "comment";
    STRING = "string";
    SYMBOL = "symbol";
    ATOM = "atom";
    NUMBER = "number";
    BRACKET = "bracket";
    INDENT_WORD_SKIP = 2;
    keywords = makeKeywords("\u03BB case-lambda call/cc class cond-expand define-class define-values exit-handler field import inherit init-field interface let*-values let-values let/ec mixin opt-lambda override protect provide public rename require require-for-syntax syntax syntax-case syntax-error unit/sig unless when with-syntax and begin call-with-current-continuation call-with-input-file call-with-output-file case cond define define-syntax define-macro defmacro delay do dynamic-wind else for-each if lambda let let* let-syntax letrec letrec-syntax map or syntax-rules abs acos angle append apply asin assoc assq assv atan boolean? caar cadr call-with-input-file call-with-output-file call-with-values car cdddar cddddr cdr ceiling char->integer char-alphabetic? char-ci<=? char-ci<? char-ci=? char-ci>=? char-ci>? char-downcase char-lower-case? char-numeric? char-ready? char-upcase char-upper-case? char-whitespace? char<=? char<? char=? char>=? char>? char? close-input-port close-output-port complex? cons cos current-input-port current-output-port denominator display eof-object? eq? equal? eqv? eval even? exact->inexact exact? exp expt #f floor force gcd imag-part inexact->exact inexact? input-port? integer->char integer? interaction-environment lcm length list list->string list->vector list-ref list-tail list? load log magnitude make-polar make-rectangular make-string make-vector max member memq memv min modulo negative? newline not null-environment null? number->string number? numerator odd? open-input-file open-output-file output-port? pair? peek-char port? positive? procedure? quasiquote quote quotient rational? rationalize read read-char real-part real? remainder reverse round scheme-report-environment set! set-car! set-cdr! sin sqrt string string->list string->number string->symbol string-append string-ci<=? string-ci<? string-ci=? string-ci>=? string-ci>? string-copy string-fill! string-length string-ref string-set! string<=? string<? string=? string>=? string>? string? substring symbol->string symbol? #t tan transcript-off transcript-on truncate values vector vector->list vector-fill! vector-length vector-ref vector-set! with-input-from-file with-output-to-file write write-char zero?");
    indentKeys = makeKeywords("define let letrec let* lambda define-macro defmacro let-syntax letrec-syntax let-values let*-values define-syntax syntax-rules define-values when unless");
    binaryMatcher = new RegExp(/^(?:[-+]i|[-+][01]+#*(?:\/[01]+#*)?i|[-+]?[01]+#*(?:\/[01]+#*)?@[-+]?[01]+#*(?:\/[01]+#*)?|[-+]?[01]+#*(?:\/[01]+#*)?[-+](?:[01]+#*(?:\/[01]+#*)?)?i|[-+]?[01]+#*(?:\/[01]+#*)?)(?=[()\s;"]|$)/i);
    octalMatcher = new RegExp(/^(?:[-+]i|[-+][0-7]+#*(?:\/[0-7]+#*)?i|[-+]?[0-7]+#*(?:\/[0-7]+#*)?@[-+]?[0-7]+#*(?:\/[0-7]+#*)?|[-+]?[0-7]+#*(?:\/[0-7]+#*)?[-+](?:[0-7]+#*(?:\/[0-7]+#*)?)?i|[-+]?[0-7]+#*(?:\/[0-7]+#*)?)(?=[()\s;"]|$)/i);
    hexMatcher = new RegExp(/^(?:[-+]i|[-+][\da-f]+#*(?:\/[\da-f]+#*)?i|[-+]?[\da-f]+#*(?:\/[\da-f]+#*)?@[-+]?[\da-f]+#*(?:\/[\da-f]+#*)?|[-+]?[\da-f]+#*(?:\/[\da-f]+#*)?[-+](?:[\da-f]+#*(?:\/[\da-f]+#*)?)?i|[-+]?[\da-f]+#*(?:\/[\da-f]+#*)?)(?=[()\s;"]|$)/i);
    decimalMatcher = new RegExp(/^(?:[-+]i|[-+](?:(?:(?:\d+#+\.?#*|\d+\.\d*#*|\.\d+#*|\d+)(?:[esfdl][-+]?\d+)?)|\d+#*\/\d+#*)i|[-+]?(?:(?:(?:\d+#+\.?#*|\d+\.\d*#*|\.\d+#*|\d+)(?:[esfdl][-+]?\d+)?)|\d+#*\/\d+#*)@[-+]?(?:(?:(?:\d+#+\.?#*|\d+\.\d*#*|\.\d+#*|\d+)(?:[esfdl][-+]?\d+)?)|\d+#*\/\d+#*)|[-+]?(?:(?:(?:\d+#+\.?#*|\d+\.\d*#*|\.\d+#*|\d+)(?:[esfdl][-+]?\d+)?)|\d+#*\/\d+#*)[-+](?:(?:(?:\d+#+\.?#*|\d+\.\d*#*|\.\d+#*|\d+)(?:[esfdl][-+]?\d+)?)|\d+#*\/\d+#*)?i|(?:(?:(?:\d+#+\.?#*|\d+\.\d*#*|\.\d+#*|\d+)(?:[esfdl][-+]?\d+)?)|\d+#*\/\d+#*))(?=[()\s;"]|$)/i);
    scheme = {
      name: "scheme",
      startState: function() {
        return {
          indentStack: null,
          indentation: 0,
          mode: false,
          sExprComment: false,
          sExprQuote: false
        };
      },
      token: function(stream, state) {
        if (state.indentStack == null && stream.sol()) {
          state.indentation = stream.indentation();
        }
        if (stream.eatSpace()) {
          return null;
        }
        var returnType = null;
        switch (state.mode) {
          case "string":
            processEscapedSequence(stream, {
              token: '"',
              state
            });
            returnType = STRING;
            break;
          case "symbol":
            processEscapedSequence(stream, {
              token: "|",
              state
            });
            returnType = SYMBOL;
            break;
          case "comment":
            var next, maybeEnd = false;
            while ((next = stream.next()) != null) {
              if (next == "#" && maybeEnd) {
                state.mode = false;
                break;
              }
              maybeEnd = next == "|";
            }
            returnType = COMMENT;
            break;
          case "s-expr-comment":
            state.mode = false;
            if (stream.peek() == "(" || stream.peek() == "[") {
              state.sExprComment = 0;
            } else {
              stream.eatWhile(/[^\s\(\)\[\]]/);
              returnType = COMMENT;
              break;
            }
          default:
            var ch = stream.next();
            if (ch == '"') {
              state.mode = "string";
              returnType = STRING;
            } else if (ch == "'") {
              if (stream.peek() == "(" || stream.peek() == "[") {
                if (typeof state.sExprQuote != "number") {
                  state.sExprQuote = 0;
                }
                returnType = ATOM;
              } else {
                stream.eatWhile(/[\w_\-!$%&*+\.\/:<=>?@\^~]/);
                returnType = ATOM;
              }
            } else if (ch == "|") {
              state.mode = "symbol";
              returnType = SYMBOL;
            } else if (ch == "#") {
              if (stream.eat("|")) {
                state.mode = "comment";
                returnType = COMMENT;
              } else if (stream.eat(/[tf]/i)) {
                returnType = ATOM;
              } else if (stream.eat(";")) {
                state.mode = "s-expr-comment";
                returnType = COMMENT;
              } else {
                var numTest = null, hasExactness = false, hasRadix = true;
                if (stream.eat(/[ei]/i)) {
                  hasExactness = true;
                } else {
                  stream.backUp(1);
                }
                if (stream.match(/^#b/i)) {
                  numTest = isBinaryNumber;
                } else if (stream.match(/^#o/i)) {
                  numTest = isOctalNumber;
                } else if (stream.match(/^#x/i)) {
                  numTest = isHexNumber;
                } else if (stream.match(/^#d/i)) {
                  numTest = isDecimalNumber;
                } else if (stream.match(/^[-+0-9.]/, false)) {
                  hasRadix = false;
                  numTest = isDecimalNumber;
                } else if (!hasExactness) {
                  stream.eat("#");
                }
                if (numTest != null) {
                  if (hasRadix && !hasExactness) {
                    stream.match(/^#[ei]/i);
                  }
                  if (numTest(stream))
                    returnType = NUMBER;
                }
              }
            } else if (/^[-+0-9.]/.test(ch) && isDecimalNumber(stream, true)) {
              returnType = NUMBER;
            } else if (ch == ";") {
              stream.skipToEnd();
              returnType = COMMENT;
            } else if (ch == "(" || ch == "[") {
              var keyWord = "";
              var indentTemp = stream.column(), letter;
              while ((letter = stream.eat(/[^\s\(\[\;\)\]]/)) != null) {
                keyWord += letter;
              }
              if (keyWord.length > 0 && indentKeys.propertyIsEnumerable(keyWord)) {
                pushStack(state, indentTemp + INDENT_WORD_SKIP, ch);
              } else {
                stream.eatSpace();
                if (stream.eol() || stream.peek() == ";") {
                  pushStack(state, indentTemp + 1, ch);
                } else {
                  pushStack(state, indentTemp + stream.current().length, ch);
                }
              }
              stream.backUp(stream.current().length - 1);
              if (typeof state.sExprComment == "number")
                state.sExprComment++;
              if (typeof state.sExprQuote == "number")
                state.sExprQuote++;
              returnType = BRACKET;
            } else if (ch == ")" || ch == "]") {
              returnType = BRACKET;
              if (state.indentStack != null && state.indentStack.type == (ch == ")" ? "(" : "[")) {
                popStack(state);
                if (typeof state.sExprComment == "number") {
                  if (--state.sExprComment == 0) {
                    returnType = COMMENT;
                    state.sExprComment = false;
                  }
                }
                if (typeof state.sExprQuote == "number") {
                  if (--state.sExprQuote == 0) {
                    returnType = ATOM;
                    state.sExprQuote = false;
                  }
                }
              }
            } else {
              stream.eatWhile(/[\w_\-!$%&*+\.\/:<=>?@\^~]/);
              if (keywords && keywords.propertyIsEnumerable(stream.current())) {
                returnType = BUILTIN;
              } else
                returnType = "variable";
            }
        }
        return typeof state.sExprComment == "number" ? COMMENT : typeof state.sExprQuote == "number" ? ATOM : returnType;
      },
      indent: function(state) {
        if (state.indentStack == null)
          return state.indentation;
        return state.indentStack.indent;
      },
      languageData: {
        closeBrackets: { brackets: ["(", "[", "{", '"'] },
        commentTokens: { line: ";;" }
      }
    };
  }
});
init_scheme();
export {
  scheme
};
//# sourceMappingURL=/build/_shared/scheme-ZMWB77L7.js.map
