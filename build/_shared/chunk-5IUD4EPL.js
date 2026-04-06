import {
  __esm
} from "/build/_shared/chunk-CGOEG7L2.js";

// ../../node_modules/@codemirror/legacy-modes/mode/javascript.js
function mkJavaScript(parserConfig) {
  var statementIndent = parserConfig.statementIndent;
  var jsonldMode = parserConfig.jsonld;
  var jsonMode = parserConfig.json || jsonldMode;
  var isTS = parserConfig.typescript;
  var wordRE = parserConfig.wordCharacters || /[\w$\xa1-\uffff]/;
  var keywords = function() {
    function kw(type2) {
      return { type: type2, style: "keyword" };
    }
    var A = kw("keyword a"), B = kw("keyword b"), C = kw("keyword c"), D = kw("keyword d");
    var operator = kw("operator"), atom = { type: "atom", style: "atom" };
    return {
      "if": kw("if"),
      "while": A,
      "with": A,
      "else": B,
      "do": B,
      "try": B,
      "finally": B,
      "return": D,
      "break": D,
      "continue": D,
      "new": kw("new"),
      "delete": C,
      "void": C,
      "throw": C,
      "debugger": kw("debugger"),
      "var": kw("var"),
      "const": kw("var"),
      "let": kw("var"),
      "function": kw("function"),
      "catch": kw("catch"),
      "for": kw("for"),
      "switch": kw("switch"),
      "case": kw("case"),
      "default": kw("default"),
      "in": operator,
      "typeof": operator,
      "instanceof": operator,
      "true": atom,
      "false": atom,
      "null": atom,
      "undefined": atom,
      "NaN": atom,
      "Infinity": atom,
      "this": kw("this"),
      "class": kw("class"),
      "super": kw("atom"),
      "yield": C,
      "export": kw("export"),
      "import": kw("import"),
      "extends": C,
      "await": C
    };
  }();
  var isOperatorChar = /[+\-*&%=<>!?|~^@]/;
  var isJsonldKeyword = /^@(context|id|value|language|type|container|list|set|reverse|index|base|vocab|graph)"/;
  function readRegexp(stream) {
    var escaped = false, next, inSet = false;
    while ((next = stream.next()) != null) {
      if (!escaped) {
        if (next == "/" && !inSet)
          return;
        if (next == "[")
          inSet = true;
        else if (inSet && next == "]")
          inSet = false;
      }
      escaped = !escaped && next == "\\";
    }
  }
  var type, content;
  function ret(tp, style, cont2) {
    type = tp;
    content = cont2;
    return style;
  }
  function tokenBase(stream, state) {
    var ch = stream.next();
    if (ch == '"' || ch == "'") {
      state.tokenize = tokenString(ch);
      return state.tokenize(stream, state);
    } else if (ch == "." && stream.match(/^\d[\d_]*(?:[eE][+\-]?[\d_]+)?/)) {
      return ret("number", "number");
    } else if (ch == "." && stream.match("..")) {
      return ret("spread", "meta");
    } else if (/[\[\]{}\(\),;\:\.]/.test(ch)) {
      return ret(ch);
    } else if (ch == "=" && stream.eat(">")) {
      return ret("=>", "operator");
    } else if (ch == "0" && stream.match(/^(?:x[\dA-Fa-f_]+|o[0-7_]+|b[01_]+)n?/)) {
      return ret("number", "number");
    } else if (/\d/.test(ch)) {
      stream.match(/^[\d_]*(?:n|(?:\.[\d_]*)?(?:[eE][+\-]?[\d_]+)?)?/);
      return ret("number", "number");
    } else if (ch == "/") {
      if (stream.eat("*")) {
        state.tokenize = tokenComment;
        return tokenComment(stream, state);
      } else if (stream.eat("/")) {
        stream.skipToEnd();
        return ret("comment", "comment");
      } else if (expressionAllowed(stream, state, 1)) {
        readRegexp(stream);
        stream.match(/^\b(([gimyus])(?![gimyus]*\2))+\b/);
        return ret("regexp", "string.special");
      } else {
        stream.eat("=");
        return ret("operator", "operator", stream.current());
      }
    } else if (ch == "`") {
      state.tokenize = tokenQuasi;
      return tokenQuasi(stream, state);
    } else if (ch == "#" && stream.peek() == "!") {
      stream.skipToEnd();
      return ret("meta", "meta");
    } else if (ch == "#" && stream.eatWhile(wordRE)) {
      return ret("variable", "property");
    } else if (ch == "<" && stream.match("!--") || ch == "-" && stream.match("->") && !/\S/.test(stream.string.slice(0, stream.start))) {
      stream.skipToEnd();
      return ret("comment", "comment");
    } else if (isOperatorChar.test(ch)) {
      if (ch != ">" || !state.lexical || state.lexical.type != ">") {
        if (stream.eat("=")) {
          if (ch == "!" || ch == "=")
            stream.eat("=");
        } else if (/[<>*+\-|&?]/.test(ch)) {
          stream.eat(ch);
          if (ch == ">")
            stream.eat(ch);
        }
      }
      if (ch == "?" && stream.eat("."))
        return ret(".");
      return ret("operator", "operator", stream.current());
    } else if (wordRE.test(ch)) {
      stream.eatWhile(wordRE);
      var word = stream.current();
      if (state.lastType != ".") {
        if (keywords.propertyIsEnumerable(word)) {
          var kw = keywords[word];
          return ret(kw.type, kw.style, word);
        }
        if (word == "async" && stream.match(/^(\s|\/\*([^*]|\*(?!\/))*?\*\/)*[\[\(\w]/, false))
          return ret("async", "keyword", word);
      }
      return ret("variable", "variable", word);
    }
  }
  function tokenString(quote) {
    return function(stream, state) {
      var escaped = false, next;
      if (jsonldMode && stream.peek() == "@" && stream.match(isJsonldKeyword)) {
        state.tokenize = tokenBase;
        return ret("jsonld-keyword", "meta");
      }
      while ((next = stream.next()) != null) {
        if (next == quote && !escaped)
          break;
        escaped = !escaped && next == "\\";
      }
      if (!escaped)
        state.tokenize = tokenBase;
      return ret("string", "string");
    };
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
    return ret("comment", "comment");
  }
  function tokenQuasi(stream, state) {
    var escaped = false, next;
    while ((next = stream.next()) != null) {
      if (!escaped && (next == "`" || next == "$" && stream.eat("{"))) {
        state.tokenize = tokenBase;
        break;
      }
      escaped = !escaped && next == "\\";
    }
    return ret("quasi", "string.special", stream.current());
  }
  var brackets = "([{}])";
  function findFatArrow(stream, state) {
    if (state.fatArrowAt)
      state.fatArrowAt = null;
    var arrow = stream.string.indexOf("=>", stream.start);
    if (arrow < 0)
      return;
    if (isTS) {
      var m = /:\s*(?:\w+(?:<[^>]*>|\[\])?|\{[^}]*\})\s*$/.exec(stream.string.slice(stream.start, arrow));
      if (m)
        arrow = m.index;
    }
    var depth = 0, sawSomething = false;
    for (var pos = arrow - 1; pos >= 0; --pos) {
      var ch = stream.string.charAt(pos);
      var bracket = brackets.indexOf(ch);
      if (bracket >= 0 && bracket < 3) {
        if (!depth) {
          ++pos;
          break;
        }
        if (--depth == 0) {
          if (ch == "(")
            sawSomething = true;
          break;
        }
      } else if (bracket >= 3 && bracket < 6) {
        ++depth;
      } else if (wordRE.test(ch)) {
        sawSomething = true;
      } else if (/["'\/`]/.test(ch)) {
        for (; ; --pos) {
          if (pos == 0)
            return;
          var next = stream.string.charAt(pos - 1);
          if (next == ch && stream.string.charAt(pos - 2) != "\\") {
            pos--;
            break;
          }
        }
      } else if (sawSomething && !depth) {
        ++pos;
        break;
      }
    }
    if (sawSomething && !depth)
      state.fatArrowAt = pos;
  }
  var atomicTypes = {
    "atom": true,
    "number": true,
    "variable": true,
    "string": true,
    "regexp": true,
    "this": true,
    "import": true,
    "jsonld-keyword": true
  };
  function JSLexical(indented, column, type2, align, prev, info) {
    this.indented = indented;
    this.column = column;
    this.type = type2;
    this.prev = prev;
    this.info = info;
    if (align != null)
      this.align = align;
  }
  function inScope(state, varname) {
    for (var v = state.localVars; v; v = v.next)
      if (v.name == varname)
        return true;
    for (var cx2 = state.context; cx2; cx2 = cx2.prev) {
      for (var v = cx2.vars; v; v = v.next)
        if (v.name == varname)
          return true;
    }
  }
  function parseJS(state, style, type2, content2, stream) {
    var cc = state.cc;
    cx.state = state;
    cx.stream = stream;
    cx.marked = null;
    cx.cc = cc;
    cx.style = style;
    if (!state.lexical.hasOwnProperty("align"))
      state.lexical.align = true;
    while (true) {
      var combinator = cc.length ? cc.pop() : jsonMode ? expression : statement;
      if (combinator(type2, content2)) {
        while (cc.length && cc[cc.length - 1].lex)
          cc.pop()();
        if (cx.marked)
          return cx.marked;
        if (type2 == "variable" && inScope(state, content2))
          return "variableName.local";
        return style;
      }
    }
  }
  var cx = { state: null, column: null, marked: null, cc: null };
  function pass() {
    for (var i = arguments.length - 1; i >= 0; i--)
      cx.cc.push(arguments[i]);
  }
  function cont() {
    pass.apply(null, arguments);
    return true;
  }
  function inList(name, list) {
    for (var v = list; v; v = v.next)
      if (v.name == name)
        return true;
    return false;
  }
  function register(varname) {
    var state = cx.state;
    cx.marked = "def";
    if (state.context) {
      if (state.lexical.info == "var" && state.context && state.context.block) {
        var newContext = registerVarScoped(varname, state.context);
        if (newContext != null) {
          state.context = newContext;
          return;
        }
      } else if (!inList(varname, state.localVars)) {
        state.localVars = new Var(varname, state.localVars);
        return;
      }
    }
    if (parserConfig.globalVars && !inList(varname, state.globalVars))
      state.globalVars = new Var(varname, state.globalVars);
  }
  function registerVarScoped(varname, context) {
    if (!context) {
      return null;
    } else if (context.block) {
      var inner = registerVarScoped(varname, context.prev);
      if (!inner)
        return null;
      if (inner == context.prev)
        return context;
      return new Context(inner, context.vars, true);
    } else if (inList(varname, context.vars)) {
      return context;
    } else {
      return new Context(context.prev, new Var(varname, context.vars), false);
    }
  }
  function isModifier(name) {
    return name == "public" || name == "private" || name == "protected" || name == "abstract" || name == "readonly";
  }
  function Context(prev, vars, block2) {
    this.prev = prev;
    this.vars = vars;
    this.block = block2;
  }
  function Var(name, next) {
    this.name = name;
    this.next = next;
  }
  var defaultVars = new Var("this", new Var("arguments", null));
  function pushcontext() {
    cx.state.context = new Context(cx.state.context, cx.state.localVars, false);
    cx.state.localVars = defaultVars;
  }
  function pushblockcontext() {
    cx.state.context = new Context(cx.state.context, cx.state.localVars, true);
    cx.state.localVars = null;
  }
  pushcontext.lex = pushblockcontext.lex = true;
  function popcontext() {
    cx.state.localVars = cx.state.context.vars;
    cx.state.context = cx.state.context.prev;
  }
  popcontext.lex = true;
  function pushlex(type2, info) {
    var result = function() {
      var state = cx.state, indent = state.indented;
      if (state.lexical.type == "stat")
        indent = state.lexical.indented;
      else
        for (var outer = state.lexical; outer && outer.type == ")" && outer.align; outer = outer.prev)
          indent = outer.indented;
      state.lexical = new JSLexical(indent, cx.stream.column(), type2, null, state.lexical, info);
    };
    result.lex = true;
    return result;
  }
  function poplex() {
    var state = cx.state;
    if (state.lexical.prev) {
      if (state.lexical.type == ")")
        state.indented = state.lexical.indented;
      state.lexical = state.lexical.prev;
    }
  }
  poplex.lex = true;
  function expect(wanted) {
    function exp(type2) {
      if (type2 == wanted)
        return cont();
      else if (wanted == ";" || type2 == "}" || type2 == ")" || type2 == "]")
        return pass();
      else
        return cont(exp);
    }
    ;
    return exp;
  }
  function statement(type2, value) {
    if (type2 == "var")
      return cont(pushlex("vardef", value), vardef, expect(";"), poplex);
    if (type2 == "keyword a")
      return cont(pushlex("form"), parenExpr, statement, poplex);
    if (type2 == "keyword b")
      return cont(pushlex("form"), statement, poplex);
    if (type2 == "keyword d")
      return cx.stream.match(/^\s*$/, false) ? cont() : cont(pushlex("stat"), maybeexpression, expect(";"), poplex);
    if (type2 == "debugger")
      return cont(expect(";"));
    if (type2 == "{")
      return cont(pushlex("}"), pushblockcontext, block, poplex, popcontext);
    if (type2 == ";")
      return cont();
    if (type2 == "if") {
      if (cx.state.lexical.info == "else" && cx.state.cc[cx.state.cc.length - 1] == poplex)
        cx.state.cc.pop()();
      return cont(pushlex("form"), parenExpr, statement, poplex, maybeelse);
    }
    if (type2 == "function")
      return cont(functiondef);
    if (type2 == "for")
      return cont(pushlex("form"), pushblockcontext, forspec, statement, popcontext, poplex);
    if (type2 == "class" || isTS && value == "interface") {
      cx.marked = "keyword";
      return cont(pushlex("form", type2 == "class" ? type2 : value), className, poplex);
    }
    if (type2 == "variable") {
      if (isTS && value == "declare") {
        cx.marked = "keyword";
        return cont(statement);
      } else if (isTS && (value == "module" || value == "enum" || value == "type") && cx.stream.match(/^\s*\w/, false)) {
        cx.marked = "keyword";
        if (value == "enum")
          return cont(enumdef);
        else if (value == "type")
          return cont(typename, expect("operator"), typeexpr, expect(";"));
        else
          return cont(pushlex("form"), pattern, expect("{"), pushlex("}"), block, poplex, poplex);
      } else if (isTS && value == "namespace") {
        cx.marked = "keyword";
        return cont(pushlex("form"), expression, statement, poplex);
      } else if (isTS && value == "abstract") {
        cx.marked = "keyword";
        return cont(statement);
      } else {
        return cont(pushlex("stat"), maybelabel);
      }
    }
    if (type2 == "switch")
      return cont(
        pushlex("form"),
        parenExpr,
        expect("{"),
        pushlex("}", "switch"),
        pushblockcontext,
        block,
        poplex,
        poplex,
        popcontext
      );
    if (type2 == "case")
      return cont(expression, expect(":"));
    if (type2 == "default")
      return cont(expect(":"));
    if (type2 == "catch")
      return cont(pushlex("form"), pushcontext, maybeCatchBinding, statement, poplex, popcontext);
    if (type2 == "export")
      return cont(pushlex("stat"), afterExport, poplex);
    if (type2 == "import")
      return cont(pushlex("stat"), afterImport, poplex);
    if (type2 == "async")
      return cont(statement);
    if (value == "@")
      return cont(expression, statement);
    return pass(pushlex("stat"), expression, expect(";"), poplex);
  }
  function maybeCatchBinding(type2) {
    if (type2 == "(")
      return cont(funarg, expect(")"));
  }
  function expression(type2, value) {
    return expressionInner(type2, value, false);
  }
  function expressionNoComma(type2, value) {
    return expressionInner(type2, value, true);
  }
  function parenExpr(type2) {
    if (type2 != "(")
      return pass();
    return cont(pushlex(")"), maybeexpression, expect(")"), poplex);
  }
  function expressionInner(type2, value, noComma) {
    if (cx.state.fatArrowAt == cx.stream.start) {
      var body = noComma ? arrowBodyNoComma : arrowBody;
      if (type2 == "(")
        return cont(pushcontext, pushlex(")"), commasep(funarg, ")"), poplex, expect("=>"), body, popcontext);
      else if (type2 == "variable")
        return pass(pushcontext, pattern, expect("=>"), body, popcontext);
    }
    var maybeop = noComma ? maybeoperatorNoComma : maybeoperatorComma;
    if (atomicTypes.hasOwnProperty(type2))
      return cont(maybeop);
    if (type2 == "function")
      return cont(functiondef, maybeop);
    if (type2 == "class" || isTS && value == "interface") {
      cx.marked = "keyword";
      return cont(pushlex("form"), classExpression, poplex);
    }
    if (type2 == "keyword c" || type2 == "async")
      return cont(noComma ? expressionNoComma : expression);
    if (type2 == "(")
      return cont(pushlex(")"), maybeexpression, expect(")"), poplex, maybeop);
    if (type2 == "operator" || type2 == "spread")
      return cont(noComma ? expressionNoComma : expression);
    if (type2 == "[")
      return cont(pushlex("]"), arrayLiteral, poplex, maybeop);
    if (type2 == "{")
      return contCommasep(objprop, "}", null, maybeop);
    if (type2 == "quasi")
      return pass(quasi, maybeop);
    if (type2 == "new")
      return cont(maybeTarget(noComma));
    return cont();
  }
  function maybeexpression(type2) {
    if (type2.match(/[;\}\)\],]/))
      return pass();
    return pass(expression);
  }
  function maybeoperatorComma(type2, value) {
    if (type2 == ",")
      return cont(maybeexpression);
    return maybeoperatorNoComma(type2, value, false);
  }
  function maybeoperatorNoComma(type2, value, noComma) {
    var me = noComma == false ? maybeoperatorComma : maybeoperatorNoComma;
    var expr = noComma == false ? expression : expressionNoComma;
    if (type2 == "=>")
      return cont(pushcontext, noComma ? arrowBodyNoComma : arrowBody, popcontext);
    if (type2 == "operator") {
      if (/\+\+|--/.test(value) || isTS && value == "!")
        return cont(me);
      if (isTS && value == "<" && cx.stream.match(/^([^<>]|<[^<>]*>)*>\s*\(/, false))
        return cont(pushlex(">"), commasep(typeexpr, ">"), poplex, me);
      if (value == "?")
        return cont(expression, expect(":"), expr);
      return cont(expr);
    }
    if (type2 == "quasi") {
      return pass(quasi, me);
    }
    if (type2 == ";")
      return;
    if (type2 == "(")
      return contCommasep(expressionNoComma, ")", "call", me);
    if (type2 == ".")
      return cont(property, me);
    if (type2 == "[")
      return cont(pushlex("]"), maybeexpression, expect("]"), poplex, me);
    if (isTS && value == "as") {
      cx.marked = "keyword";
      return cont(typeexpr, me);
    }
    if (type2 == "regexp") {
      cx.state.lastType = cx.marked = "operator";
      cx.stream.backUp(cx.stream.pos - cx.stream.start - 1);
      return cont(expr);
    }
  }
  function quasi(type2, value) {
    if (type2 != "quasi")
      return pass();
    if (value.slice(value.length - 2) != "${")
      return cont(quasi);
    return cont(maybeexpression, continueQuasi);
  }
  function continueQuasi(type2) {
    if (type2 == "}") {
      cx.marked = "string.special";
      cx.state.tokenize = tokenQuasi;
      return cont(quasi);
    }
  }
  function arrowBody(type2) {
    findFatArrow(cx.stream, cx.state);
    return pass(type2 == "{" ? statement : expression);
  }
  function arrowBodyNoComma(type2) {
    findFatArrow(cx.stream, cx.state);
    return pass(type2 == "{" ? statement : expressionNoComma);
  }
  function maybeTarget(noComma) {
    return function(type2) {
      if (type2 == ".")
        return cont(noComma ? targetNoComma : target);
      else if (type2 == "variable" && isTS)
        return cont(maybeTypeArgs, noComma ? maybeoperatorNoComma : maybeoperatorComma);
      else
        return pass(noComma ? expressionNoComma : expression);
    };
  }
  function target(_, value) {
    if (value == "target") {
      cx.marked = "keyword";
      return cont(maybeoperatorComma);
    }
  }
  function targetNoComma(_, value) {
    if (value == "target") {
      cx.marked = "keyword";
      return cont(maybeoperatorNoComma);
    }
  }
  function maybelabel(type2) {
    if (type2 == ":")
      return cont(poplex, statement);
    return pass(maybeoperatorComma, expect(";"), poplex);
  }
  function property(type2) {
    if (type2 == "variable") {
      cx.marked = "property";
      return cont();
    }
  }
  function objprop(type2, value) {
    if (type2 == "async") {
      cx.marked = "property";
      return cont(objprop);
    } else if (type2 == "variable" || cx.style == "keyword") {
      cx.marked = "property";
      if (value == "get" || value == "set")
        return cont(getterSetter);
      var m;
      if (isTS && cx.state.fatArrowAt == cx.stream.start && (m = cx.stream.match(/^\s*:\s*/, false)))
        cx.state.fatArrowAt = cx.stream.pos + m[0].length;
      return cont(afterprop);
    } else if (type2 == "number" || type2 == "string") {
      cx.marked = jsonldMode ? "property" : cx.style + " property";
      return cont(afterprop);
    } else if (type2 == "jsonld-keyword") {
      return cont(afterprop);
    } else if (isTS && isModifier(value)) {
      cx.marked = "keyword";
      return cont(objprop);
    } else if (type2 == "[") {
      return cont(expression, maybetype, expect("]"), afterprop);
    } else if (type2 == "spread") {
      return cont(expressionNoComma, afterprop);
    } else if (value == "*") {
      cx.marked = "keyword";
      return cont(objprop);
    } else if (type2 == ":") {
      return pass(afterprop);
    }
  }
  function getterSetter(type2) {
    if (type2 != "variable")
      return pass(afterprop);
    cx.marked = "property";
    return cont(functiondef);
  }
  function afterprop(type2) {
    if (type2 == ":")
      return cont(expressionNoComma);
    if (type2 == "(")
      return pass(functiondef);
  }
  function commasep(what, end, sep) {
    function proceed(type2, value) {
      if (sep ? sep.indexOf(type2) > -1 : type2 == ",") {
        var lex = cx.state.lexical;
        if (lex.info == "call")
          lex.pos = (lex.pos || 0) + 1;
        return cont(function(type3, value2) {
          if (type3 == end || value2 == end)
            return pass();
          return pass(what);
        }, proceed);
      }
      if (type2 == end || value == end)
        return cont();
      if (sep && sep.indexOf(";") > -1)
        return pass(what);
      return cont(expect(end));
    }
    return function(type2, value) {
      if (type2 == end || value == end)
        return cont();
      return pass(what, proceed);
    };
  }
  function contCommasep(what, end, info) {
    for (var i = 3; i < arguments.length; i++)
      cx.cc.push(arguments[i]);
    return cont(pushlex(end, info), commasep(what, end), poplex);
  }
  function block(type2) {
    if (type2 == "}")
      return cont();
    return pass(statement, block);
  }
  function maybetype(type2, value) {
    if (isTS) {
      if (type2 == ":")
        return cont(typeexpr);
      if (value == "?")
        return cont(maybetype);
    }
  }
  function maybetypeOrIn(type2, value) {
    if (isTS && (type2 == ":" || value == "in"))
      return cont(typeexpr);
  }
  function mayberettype(type2) {
    if (isTS && type2 == ":") {
      if (cx.stream.match(/^\s*\w+\s+is\b/, false))
        return cont(expression, isKW, typeexpr);
      else
        return cont(typeexpr);
    }
  }
  function isKW(_, value) {
    if (value == "is") {
      cx.marked = "keyword";
      return cont();
    }
  }
  function typeexpr(type2, value) {
    if (value == "keyof" || value == "typeof" || value == "infer" || value == "readonly") {
      cx.marked = "keyword";
      return cont(value == "typeof" ? expressionNoComma : typeexpr);
    }
    if (type2 == "variable" || value == "void") {
      cx.marked = "type";
      return cont(afterType);
    }
    if (value == "|" || value == "&")
      return cont(typeexpr);
    if (type2 == "string" || type2 == "number" || type2 == "atom")
      return cont(afterType);
    if (type2 == "[")
      return cont(pushlex("]"), commasep(typeexpr, "]", ","), poplex, afterType);
    if (type2 == "{")
      return cont(pushlex("}"), typeprops, poplex, afterType);
    if (type2 == "(")
      return cont(commasep(typearg, ")"), maybeReturnType, afterType);
    if (type2 == "<")
      return cont(commasep(typeexpr, ">"), typeexpr);
    if (type2 == "quasi")
      return pass(quasiType, afterType);
  }
  function maybeReturnType(type2) {
    if (type2 == "=>")
      return cont(typeexpr);
  }
  function typeprops(type2) {
    if (type2.match(/[\}\)\]]/))
      return cont();
    if (type2 == "," || type2 == ";")
      return cont(typeprops);
    return pass(typeprop, typeprops);
  }
  function typeprop(type2, value) {
    if (type2 == "variable" || cx.style == "keyword") {
      cx.marked = "property";
      return cont(typeprop);
    } else if (value == "?" || type2 == "number" || type2 == "string") {
      return cont(typeprop);
    } else if (type2 == ":") {
      return cont(typeexpr);
    } else if (type2 == "[") {
      return cont(expect("variable"), maybetypeOrIn, expect("]"), typeprop);
    } else if (type2 == "(") {
      return pass(functiondecl, typeprop);
    } else if (!type2.match(/[;\}\)\],]/)) {
      return cont();
    }
  }
  function quasiType(type2, value) {
    if (type2 != "quasi")
      return pass();
    if (value.slice(value.length - 2) != "${")
      return cont(quasiType);
    return cont(typeexpr, continueQuasiType);
  }
  function continueQuasiType(type2) {
    if (type2 == "}") {
      cx.marked = "string.special";
      cx.state.tokenize = tokenQuasi;
      return cont(quasiType);
    }
  }
  function typearg(type2, value) {
    if (type2 == "variable" && cx.stream.match(/^\s*[?:]/, false) || value == "?")
      return cont(typearg);
    if (type2 == ":")
      return cont(typeexpr);
    if (type2 == "spread")
      return cont(typearg);
    return pass(typeexpr);
  }
  function afterType(type2, value) {
    if (value == "<")
      return cont(pushlex(">"), commasep(typeexpr, ">"), poplex, afterType);
    if (value == "|" || type2 == "." || value == "&")
      return cont(typeexpr);
    if (type2 == "[")
      return cont(typeexpr, expect("]"), afterType);
    if (value == "extends" || value == "implements") {
      cx.marked = "keyword";
      return cont(typeexpr);
    }
    if (value == "?")
      return cont(typeexpr, expect(":"), typeexpr);
  }
  function maybeTypeArgs(_, value) {
    if (value == "<")
      return cont(pushlex(">"), commasep(typeexpr, ">"), poplex, afterType);
  }
  function typeparam() {
    return pass(typeexpr, maybeTypeDefault);
  }
  function maybeTypeDefault(_, value) {
    if (value == "=")
      return cont(typeexpr);
  }
  function vardef(_, value) {
    if (value == "enum") {
      cx.marked = "keyword";
      return cont(enumdef);
    }
    return pass(pattern, maybetype, maybeAssign, vardefCont);
  }
  function pattern(type2, value) {
    if (isTS && isModifier(value)) {
      cx.marked = "keyword";
      return cont(pattern);
    }
    if (type2 == "variable") {
      register(value);
      return cont();
    }
    if (type2 == "spread")
      return cont(pattern);
    if (type2 == "[")
      return contCommasep(eltpattern, "]");
    if (type2 == "{")
      return contCommasep(proppattern, "}");
  }
  function proppattern(type2, value) {
    if (type2 == "variable" && !cx.stream.match(/^\s*:/, false)) {
      register(value);
      return cont(maybeAssign);
    }
    if (type2 == "variable")
      cx.marked = "property";
    if (type2 == "spread")
      return cont(pattern);
    if (type2 == "}")
      return pass();
    if (type2 == "[")
      return cont(expression, expect("]"), expect(":"), proppattern);
    return cont(expect(":"), pattern, maybeAssign);
  }
  function eltpattern() {
    return pass(pattern, maybeAssign);
  }
  function maybeAssign(_type, value) {
    if (value == "=")
      return cont(expressionNoComma);
  }
  function vardefCont(type2) {
    if (type2 == ",")
      return cont(vardef);
  }
  function maybeelse(type2, value) {
    if (type2 == "keyword b" && value == "else")
      return cont(pushlex("form", "else"), statement, poplex);
  }
  function forspec(type2, value) {
    if (value == "await")
      return cont(forspec);
    if (type2 == "(")
      return cont(pushlex(")"), forspec1, poplex);
  }
  function forspec1(type2) {
    if (type2 == "var")
      return cont(vardef, forspec2);
    if (type2 == "variable")
      return cont(forspec2);
    return pass(forspec2);
  }
  function forspec2(type2, value) {
    if (type2 == ")")
      return cont();
    if (type2 == ";")
      return cont(forspec2);
    if (value == "in" || value == "of") {
      cx.marked = "keyword";
      return cont(expression, forspec2);
    }
    return pass(expression, forspec2);
  }
  function functiondef(type2, value) {
    if (value == "*") {
      cx.marked = "keyword";
      return cont(functiondef);
    }
    if (type2 == "variable") {
      register(value);
      return cont(functiondef);
    }
    if (type2 == "(")
      return cont(pushcontext, pushlex(")"), commasep(funarg, ")"), poplex, mayberettype, statement, popcontext);
    if (isTS && value == "<")
      return cont(pushlex(">"), commasep(typeparam, ">"), poplex, functiondef);
  }
  function functiondecl(type2, value) {
    if (value == "*") {
      cx.marked = "keyword";
      return cont(functiondecl);
    }
    if (type2 == "variable") {
      register(value);
      return cont(functiondecl);
    }
    if (type2 == "(")
      return cont(pushcontext, pushlex(")"), commasep(funarg, ")"), poplex, mayberettype, popcontext);
    if (isTS && value == "<")
      return cont(pushlex(">"), commasep(typeparam, ">"), poplex, functiondecl);
  }
  function typename(type2, value) {
    if (type2 == "keyword" || type2 == "variable") {
      cx.marked = "type";
      return cont(typename);
    } else if (value == "<") {
      return cont(pushlex(">"), commasep(typeparam, ">"), poplex);
    }
  }
  function funarg(type2, value) {
    if (value == "@")
      cont(expression, funarg);
    if (type2 == "spread")
      return cont(funarg);
    if (isTS && isModifier(value)) {
      cx.marked = "keyword";
      return cont(funarg);
    }
    if (isTS && type2 == "this")
      return cont(maybetype, maybeAssign);
    return pass(pattern, maybetype, maybeAssign);
  }
  function classExpression(type2, value) {
    if (type2 == "variable")
      return className(type2, value);
    return classNameAfter(type2, value);
  }
  function className(type2, value) {
    if (type2 == "variable") {
      register(value);
      return cont(classNameAfter);
    }
  }
  function classNameAfter(type2, value) {
    if (value == "<")
      return cont(pushlex(">"), commasep(typeparam, ">"), poplex, classNameAfter);
    if (value == "extends" || value == "implements" || isTS && type2 == ",") {
      if (value == "implements")
        cx.marked = "keyword";
      return cont(isTS ? typeexpr : expression, classNameAfter);
    }
    if (type2 == "{")
      return cont(pushlex("}"), classBody, poplex);
  }
  function classBody(type2, value) {
    if (type2 == "async" || type2 == "variable" && (value == "static" || value == "get" || value == "set" || isTS && isModifier(value)) && cx.stream.match(/^\s+#?[\w$\xa1-\uffff]/, false)) {
      cx.marked = "keyword";
      return cont(classBody);
    }
    if (type2 == "variable" || cx.style == "keyword") {
      cx.marked = "property";
      return cont(classfield, classBody);
    }
    if (type2 == "number" || type2 == "string")
      return cont(classfield, classBody);
    if (type2 == "[")
      return cont(expression, maybetype, expect("]"), classfield, classBody);
    if (value == "*") {
      cx.marked = "keyword";
      return cont(classBody);
    }
    if (isTS && type2 == "(")
      return pass(functiondecl, classBody);
    if (type2 == ";" || type2 == ",")
      return cont(classBody);
    if (type2 == "}")
      return cont();
    if (value == "@")
      return cont(expression, classBody);
  }
  function classfield(type2, value) {
    if (value == "!" || value == "?")
      return cont(classfield);
    if (type2 == ":")
      return cont(typeexpr, maybeAssign);
    if (value == "=")
      return cont(expressionNoComma);
    var context = cx.state.lexical.prev, isInterface = context && context.info == "interface";
    return pass(isInterface ? functiondecl : functiondef);
  }
  function afterExport(type2, value) {
    if (value == "*") {
      cx.marked = "keyword";
      return cont(maybeFrom, expect(";"));
    }
    if (value == "default") {
      cx.marked = "keyword";
      return cont(expression, expect(";"));
    }
    if (type2 == "{")
      return cont(commasep(exportField, "}"), maybeFrom, expect(";"));
    return pass(statement);
  }
  function exportField(type2, value) {
    if (value == "as") {
      cx.marked = "keyword";
      return cont(expect("variable"));
    }
    if (type2 == "variable")
      return pass(expressionNoComma, exportField);
  }
  function afterImport(type2) {
    if (type2 == "string")
      return cont();
    if (type2 == "(")
      return pass(expression);
    if (type2 == ".")
      return pass(maybeoperatorComma);
    return pass(importSpec, maybeMoreImports, maybeFrom);
  }
  function importSpec(type2, value) {
    if (type2 == "{")
      return contCommasep(importSpec, "}");
    if (type2 == "variable")
      register(value);
    if (value == "*")
      cx.marked = "keyword";
    return cont(maybeAs);
  }
  function maybeMoreImports(type2) {
    if (type2 == ",")
      return cont(importSpec, maybeMoreImports);
  }
  function maybeAs(_type, value) {
    if (value == "as") {
      cx.marked = "keyword";
      return cont(importSpec);
    }
  }
  function maybeFrom(_type, value) {
    if (value == "from") {
      cx.marked = "keyword";
      return cont(expression);
    }
  }
  function arrayLiteral(type2) {
    if (type2 == "]")
      return cont();
    return pass(commasep(expressionNoComma, "]"));
  }
  function enumdef() {
    return pass(pushlex("form"), pattern, expect("{"), pushlex("}"), commasep(enummember, "}"), poplex, poplex);
  }
  function enummember() {
    return pass(pattern, maybeAssign);
  }
  function isContinuedStatement(state, textAfter) {
    return state.lastType == "operator" || state.lastType == "," || isOperatorChar.test(textAfter.charAt(0)) || /[,.]/.test(textAfter.charAt(0));
  }
  function expressionAllowed(stream, state, backUp) {
    return state.tokenize == tokenBase && /^(?:operator|sof|keyword [bcd]|case|new|export|default|spread|[\[{}\(,;:]|=>)$/.test(state.lastType) || state.lastType == "quasi" && /\{\s*$/.test(stream.string.slice(0, stream.pos - (backUp || 0)));
  }
  return {
    name: parserConfig.name,
    startState: function(indentUnit) {
      var state = {
        tokenize: tokenBase,
        lastType: "sof",
        cc: [],
        lexical: new JSLexical(-indentUnit, 0, "block", false),
        localVars: parserConfig.localVars,
        context: parserConfig.localVars && new Context(null, null, false),
        indented: 0
      };
      if (parserConfig.globalVars && typeof parserConfig.globalVars == "object")
        state.globalVars = parserConfig.globalVars;
      return state;
    },
    token: function(stream, state) {
      if (stream.sol()) {
        if (!state.lexical.hasOwnProperty("align"))
          state.lexical.align = false;
        state.indented = stream.indentation();
        findFatArrow(stream, state);
      }
      if (state.tokenize != tokenComment && stream.eatSpace())
        return null;
      var style = state.tokenize(stream, state);
      if (type == "comment")
        return style;
      state.lastType = type == "operator" && (content == "++" || content == "--") ? "incdec" : type;
      return parseJS(state, style, type, content, stream);
    },
    indent: function(state, textAfter, cx2) {
      if (state.tokenize == tokenComment || state.tokenize == tokenQuasi)
        return null;
      if (state.tokenize != tokenBase)
        return 0;
      var firstChar = textAfter && textAfter.charAt(0), lexical = state.lexical, top;
      if (!/^\s*else\b/.test(textAfter))
        for (var i = state.cc.length - 1; i >= 0; --i) {
          var c = state.cc[i];
          if (c == poplex)
            lexical = lexical.prev;
          else if (c != maybeelse && c != popcontext)
            break;
        }
      while ((lexical.type == "stat" || lexical.type == "form") && (firstChar == "}" || (top = state.cc[state.cc.length - 1]) && (top == maybeoperatorComma || top == maybeoperatorNoComma) && !/^[,\.=+\-*:?[\(]/.test(textAfter)))
        lexical = lexical.prev;
      if (statementIndent && lexical.type == ")" && lexical.prev.type == "stat")
        lexical = lexical.prev;
      var type2 = lexical.type, closing = firstChar == type2;
      if (type2 == "vardef")
        return lexical.indented + (state.lastType == "operator" || state.lastType == "," ? lexical.info.length + 1 : 0);
      else if (type2 == "form" && firstChar == "{")
        return lexical.indented;
      else if (type2 == "form")
        return lexical.indented + cx2.unit;
      else if (type2 == "stat")
        return lexical.indented + (isContinuedStatement(state, textAfter) ? statementIndent || cx2.unit : 0);
      else if (lexical.info == "switch" && !closing && parserConfig.doubleIndentSwitch != false)
        return lexical.indented + (/^(?:case|default)\b/.test(textAfter) ? cx2.unit : 2 * cx2.unit);
      else if (lexical.align)
        return lexical.column + (closing ? 0 : 1);
      else
        return lexical.indented + (closing ? 0 : cx2.unit);
    },
    languageData: {
      indentOnInput: /^\s*(?:case .*?:|default:|\{|\})$/,
      commentTokens: jsonMode ? void 0 : { line: "//", block: { open: "/*", close: "*/" } },
      closeBrackets: { brackets: ["(", "[", "{", "'", '"', "`"] },
      wordChars: "$"
    }
  };
}
var javascript, json, jsonld, typescript;
var init_javascript = __esm({
  "../../node_modules/@codemirror/legacy-modes/mode/javascript.js"() {
    javascript = mkJavaScript({ name: "javascript" });
    json = mkJavaScript({ name: "json", json: true });
    jsonld = mkJavaScript({ name: "json", jsonld: true });
    typescript = mkJavaScript({ name: "typescript", typescript: true });
  }
});

export {
  javascript,
  json,
  jsonld,
  typescript,
  init_javascript
};
//# sourceMappingURL=/build/_shared/chunk-5IUD4EPL.js.map
