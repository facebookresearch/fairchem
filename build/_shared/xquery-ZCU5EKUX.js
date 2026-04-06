import {
  __esm
} from "/build/_shared/chunk-CGOEG7L2.js";

// ../../node_modules/@codemirror/legacy-modes/mode/xquery.js
function chain(stream, state, f) {
  state.tokenize = f;
  return f(stream, state);
}
function tokenBase(stream, state) {
  var ch = stream.next(), mightBeFunction = false, isEQName = isEQNameAhead(stream);
  if (ch == "<") {
    if (stream.match("!--", true))
      return chain(stream, state, tokenXMLComment);
    if (stream.match("![CDATA", false)) {
      state.tokenize = tokenCDATA;
      return "tag";
    }
    if (stream.match("?", false)) {
      return chain(stream, state, tokenPreProcessing);
    }
    var isclose = stream.eat("/");
    stream.eatSpace();
    var tagName = "", c;
    while (c = stream.eat(/[^\s\u00a0=<>\"\'\/?]/))
      tagName += c;
    return chain(stream, state, tokenTag(tagName, isclose));
  } else if (ch == "{") {
    pushStateStack(state, { type: "codeblock" });
    return null;
  } else if (ch == "}") {
    popStateStack(state);
    return null;
  } else if (isInXmlBlock(state)) {
    if (ch == ">")
      return "tag";
    else if (ch == "/" && stream.eat(">")) {
      popStateStack(state);
      return "tag";
    } else
      return "variable";
  } else if (/\d/.test(ch)) {
    stream.match(/^\d*(?:\.\d*)?(?:E[+\-]?\d+)?/);
    return "atom";
  } else if (ch === "(" && stream.eat(":")) {
    pushStateStack(state, { type: "comment" });
    return chain(stream, state, tokenComment);
  } else if (!isEQName && (ch === '"' || ch === "'"))
    return startString(stream, state, ch);
  else if (ch === "$") {
    return chain(stream, state, tokenVariable);
  } else if (ch === ":" && stream.eat("=")) {
    return "keyword";
  } else if (ch === "(") {
    pushStateStack(state, { type: "paren" });
    return null;
  } else if (ch === ")") {
    popStateStack(state);
    return null;
  } else if (ch === "[") {
    pushStateStack(state, { type: "bracket" });
    return null;
  } else if (ch === "]") {
    popStateStack(state);
    return null;
  } else {
    var known = keywords.propertyIsEnumerable(ch) && keywords[ch];
    if (isEQName && ch === '"')
      while (stream.next() !== '"') {
      }
    if (isEQName && ch === "'")
      while (stream.next() !== "'") {
      }
    if (!known)
      stream.eatWhile(/[\w\$_-]/);
    var foundColon = stream.eat(":");
    if (!stream.eat(":") && foundColon) {
      stream.eatWhile(/[\w\$_-]/);
    }
    if (stream.match(/^[ \t]*\(/, false)) {
      mightBeFunction = true;
    }
    var word = stream.current();
    known = keywords.propertyIsEnumerable(word) && keywords[word];
    if (mightBeFunction && !known)
      known = { type: "function_call", style: "def" };
    if (isInXmlConstructor(state)) {
      popStateStack(state);
      return "variable";
    }
    if (word == "element" || word == "attribute" || known.type == "axis_specifier")
      pushStateStack(state, { type: "xmlconstructor" });
    return known ? known.style : "variable";
  }
}
function tokenComment(stream, state) {
  var maybeEnd = false, maybeNested = false, nestedCount = 0, ch;
  while (ch = stream.next()) {
    if (ch == ")" && maybeEnd) {
      if (nestedCount > 0)
        nestedCount--;
      else {
        popStateStack(state);
        break;
      }
    } else if (ch == ":" && maybeNested) {
      nestedCount++;
    }
    maybeEnd = ch == ":";
    maybeNested = ch == "(";
  }
  return "comment";
}
function tokenString(quote, f) {
  return function(stream, state) {
    var ch;
    while (ch = stream.next()) {
      if (ch == quote) {
        popStateStack(state);
        if (f)
          state.tokenize = f;
        break;
      } else if (stream.match("{", false) && isInXmlAttributeBlock(state)) {
        pushStateStack(state, { type: "codeblock" });
        state.tokenize = tokenBase;
        return "string";
      }
    }
    return "string";
  };
}
function startString(stream, state, quote, f) {
  let tokenize = tokenString(quote, f);
  pushStateStack(state, { type: "string", name: quote, tokenize });
  return chain(stream, state, tokenize);
}
function tokenVariable(stream, state) {
  var isVariableChar = /[\w\$_-]/;
  if (stream.eat('"')) {
    while (stream.next() !== '"') {
    }
    ;
    stream.eat(":");
  } else {
    stream.eatWhile(isVariableChar);
    if (!stream.match(":=", false))
      stream.eat(":");
  }
  stream.eatWhile(isVariableChar);
  state.tokenize = tokenBase;
  return "variable";
}
function tokenTag(name, isclose) {
  return function(stream, state) {
    stream.eatSpace();
    if (isclose && stream.eat(">")) {
      popStateStack(state);
      state.tokenize = tokenBase;
      return "tag";
    }
    if (!stream.eat("/"))
      pushStateStack(state, { type: "tag", name, tokenize: tokenBase });
    if (!stream.eat(">")) {
      state.tokenize = tokenAttribute;
      return "tag";
    } else {
      state.tokenize = tokenBase;
    }
    return "tag";
  };
}
function tokenAttribute(stream, state) {
  var ch = stream.next();
  if (ch == "/" && stream.eat(">")) {
    if (isInXmlAttributeBlock(state))
      popStateStack(state);
    if (isInXmlBlock(state))
      popStateStack(state);
    return "tag";
  }
  if (ch == ">") {
    if (isInXmlAttributeBlock(state))
      popStateStack(state);
    return "tag";
  }
  if (ch == "=")
    return null;
  if (ch == '"' || ch == "'")
    return startString(stream, state, ch, tokenAttribute);
  if (!isInXmlAttributeBlock(state))
    pushStateStack(state, { type: "attribute", tokenize: tokenAttribute });
  stream.eat(/[a-zA-Z_:]/);
  stream.eatWhile(/[-a-zA-Z0-9_:.]/);
  stream.eatSpace();
  if (stream.match(">", false) || stream.match("/", false)) {
    popStateStack(state);
    state.tokenize = tokenBase;
  }
  return "attribute";
}
function tokenXMLComment(stream, state) {
  var ch;
  while (ch = stream.next()) {
    if (ch == "-" && stream.match("->", true)) {
      state.tokenize = tokenBase;
      return "comment";
    }
  }
}
function tokenCDATA(stream, state) {
  var ch;
  while (ch = stream.next()) {
    if (ch == "]" && stream.match("]", true)) {
      state.tokenize = tokenBase;
      return "comment";
    }
  }
}
function tokenPreProcessing(stream, state) {
  var ch;
  while (ch = stream.next()) {
    if (ch == "?" && stream.match(">", true)) {
      state.tokenize = tokenBase;
      return "processingInstruction";
    }
  }
}
function isInXmlBlock(state) {
  return isIn(state, "tag");
}
function isInXmlAttributeBlock(state) {
  return isIn(state, "attribute");
}
function isInXmlConstructor(state) {
  return isIn(state, "xmlconstructor");
}
function isEQNameAhead(stream) {
  if (stream.current() === '"')
    return stream.match(/^[^\"]+\"\:/, false);
  else if (stream.current() === "'")
    return stream.match(/^[^\"]+\'\:/, false);
  else
    return false;
}
function isIn(state, type) {
  return state.stack.length && state.stack[state.stack.length - 1].type == type;
}
function pushStateStack(state, newState) {
  state.stack.push(newState);
}
function popStateStack(state) {
  state.stack.pop();
  var reinstateTokenize = state.stack.length && state.stack[state.stack.length - 1].tokenize;
  state.tokenize = reinstateTokenize || tokenBase;
}
var keywords, xQuery;
var init_xquery = __esm({
  "../../node_modules/@codemirror/legacy-modes/mode/xquery.js"() {
    keywords = function() {
      function kw(type) {
        return { type, style: "keyword" };
      }
      var operator = kw("operator"), atom = { type: "atom", style: "atom" }, punctuation = { type: "punctuation", style: null }, qualifier = { type: "axis_specifier", style: "qualifier" };
      var kwObj = {
        ",": punctuation
      };
      var basic = [
        "after",
        "all",
        "allowing",
        "ancestor",
        "ancestor-or-self",
        "any",
        "array",
        "as",
        "ascending",
        "at",
        "attribute",
        "base-uri",
        "before",
        "boundary-space",
        "by",
        "case",
        "cast",
        "castable",
        "catch",
        "child",
        "collation",
        "comment",
        "construction",
        "contains",
        "content",
        "context",
        "copy",
        "copy-namespaces",
        "count",
        "decimal-format",
        "declare",
        "default",
        "delete",
        "descendant",
        "descendant-or-self",
        "descending",
        "diacritics",
        "different",
        "distance",
        "document",
        "document-node",
        "element",
        "else",
        "empty",
        "empty-sequence",
        "encoding",
        "end",
        "entire",
        "every",
        "exactly",
        "except",
        "external",
        "first",
        "following",
        "following-sibling",
        "for",
        "from",
        "ftand",
        "ftnot",
        "ft-option",
        "ftor",
        "function",
        "fuzzy",
        "greatest",
        "group",
        "if",
        "import",
        "in",
        "inherit",
        "insensitive",
        "insert",
        "instance",
        "intersect",
        "into",
        "invoke",
        "is",
        "item",
        "language",
        "last",
        "lax",
        "least",
        "let",
        "levels",
        "lowercase",
        "map",
        "modify",
        "module",
        "most",
        "namespace",
        "next",
        "no",
        "node",
        "nodes",
        "no-inherit",
        "no-preserve",
        "not",
        "occurs",
        "of",
        "only",
        "option",
        "order",
        "ordered",
        "ordering",
        "paragraph",
        "paragraphs",
        "parent",
        "phrase",
        "preceding",
        "preceding-sibling",
        "preserve",
        "previous",
        "processing-instruction",
        "relationship",
        "rename",
        "replace",
        "return",
        "revalidation",
        "same",
        "satisfies",
        "schema",
        "schema-attribute",
        "schema-element",
        "score",
        "self",
        "sensitive",
        "sentence",
        "sentences",
        "sequence",
        "skip",
        "sliding",
        "some",
        "stable",
        "start",
        "stemming",
        "stop",
        "strict",
        "strip",
        "switch",
        "text",
        "then",
        "thesaurus",
        "times",
        "to",
        "transform",
        "treat",
        "try",
        "tumbling",
        "type",
        "typeswitch",
        "union",
        "unordered",
        "update",
        "updating",
        "uppercase",
        "using",
        "validate",
        "value",
        "variable",
        "version",
        "weight",
        "when",
        "where",
        "wildcards",
        "window",
        "with",
        "without",
        "word",
        "words",
        "xquery"
      ];
      for (var i = 0, l = basic.length; i < l; i++) {
        kwObj[basic[i]] = kw(basic[i]);
      }
      ;
      var types = [
        "xs:anyAtomicType",
        "xs:anySimpleType",
        "xs:anyType",
        "xs:anyURI",
        "xs:base64Binary",
        "xs:boolean",
        "xs:byte",
        "xs:date",
        "xs:dateTime",
        "xs:dateTimeStamp",
        "xs:dayTimeDuration",
        "xs:decimal",
        "xs:double",
        "xs:duration",
        "xs:ENTITIES",
        "xs:ENTITY",
        "xs:float",
        "xs:gDay",
        "xs:gMonth",
        "xs:gMonthDay",
        "xs:gYear",
        "xs:gYearMonth",
        "xs:hexBinary",
        "xs:ID",
        "xs:IDREF",
        "xs:IDREFS",
        "xs:int",
        "xs:integer",
        "xs:item",
        "xs:java",
        "xs:language",
        "xs:long",
        "xs:Name",
        "xs:NCName",
        "xs:negativeInteger",
        "xs:NMTOKEN",
        "xs:NMTOKENS",
        "xs:nonNegativeInteger",
        "xs:nonPositiveInteger",
        "xs:normalizedString",
        "xs:NOTATION",
        "xs:numeric",
        "xs:positiveInteger",
        "xs:precisionDecimal",
        "xs:QName",
        "xs:short",
        "xs:string",
        "xs:time",
        "xs:token",
        "xs:unsignedByte",
        "xs:unsignedInt",
        "xs:unsignedLong",
        "xs:unsignedShort",
        "xs:untyped",
        "xs:untypedAtomic",
        "xs:yearMonthDuration"
      ];
      for (var i = 0, l = types.length; i < l; i++) {
        kwObj[types[i]] = atom;
      }
      ;
      var operators = ["eq", "ne", "lt", "le", "gt", "ge", ":=", "=", ">", ">=", "<", "<=", ".", "|", "?", "and", "or", "div", "idiv", "mod", "*", "/", "+", "-"];
      for (var i = 0, l = operators.length; i < l; i++) {
        kwObj[operators[i]] = operator;
      }
      ;
      var axis_specifiers = [
        "self::",
        "attribute::",
        "child::",
        "descendant::",
        "descendant-or-self::",
        "parent::",
        "ancestor::",
        "ancestor-or-self::",
        "following::",
        "preceding::",
        "following-sibling::",
        "preceding-sibling::"
      ];
      for (var i = 0, l = axis_specifiers.length; i < l; i++) {
        kwObj[axis_specifiers[i]] = qualifier;
      }
      ;
      return kwObj;
    }();
    xQuery = {
      name: "xquery",
      startState: function() {
        return {
          tokenize: tokenBase,
          cc: [],
          stack: []
        };
      },
      token: function(stream, state) {
        if (stream.eatSpace())
          return null;
        var style = state.tokenize(stream, state);
        return style;
      },
      languageData: {
        commentTokens: { block: { open: "(:", close: ":)" } }
      }
    };
  }
});
init_xquery();
export {
  xQuery
};
//# sourceMappingURL=/build/_shared/xquery-ZCU5EKUX.js.map
