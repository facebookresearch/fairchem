import {
  __esm
} from "/build/_shared/chunk-CGOEG7L2.js";

// ../../node_modules/@codemirror/legacy-modes/mode/webidl.js
function wordRegexp(words) {
  return new RegExp("^((" + words.join(")|(") + "))\\b");
}
function readToken(stream, state) {
  if (stream.eatSpace())
    return null;
  if (state.inComment) {
    if (stream.match(multilineCommentsEnd)) {
      state.inComment = false;
      return "comment";
    }
    stream.skipToEnd();
    return "comment";
  }
  if (stream.match("//")) {
    stream.skipToEnd();
    return "comment";
  }
  if (stream.match(multilineComments))
    return "comment";
  if (stream.match(multilineCommentsStart)) {
    state.inComment = true;
    return "comment";
  }
  if (stream.match(/^-?[0-9\.]/, false)) {
    if (stream.match(integers) || stream.match(floats))
      return "number";
  }
  if (stream.match(strings))
    return "string";
  if (state.startDef && stream.match(identifiers))
    return "def";
  if (state.endDef && stream.match(identifiersEnd)) {
    state.endDef = false;
    return "def";
  }
  if (stream.match(keywords))
    return "keyword";
  if (stream.match(types)) {
    var lastToken = state.lastToken;
    var nextToken = (stream.match(/^\s*(.+?)\b/, false) || [])[1];
    if (lastToken === ":" || lastToken === "implements" || nextToken === "implements" || nextToken === "=") {
      return "builtin";
    } else {
      return "type";
    }
  }
  if (stream.match(builtins))
    return "builtin";
  if (stream.match(atoms))
    return "atom";
  if (stream.match(identifiers))
    return "variable";
  if (stream.match(singleOperators))
    return "operator";
  stream.next();
  return null;
}
var builtinArray, builtins, typeArray, types, keywordArray, keywords, atomArray, atoms, startDefArray, startDefs, endDefArray, endDefs, singleOperators, integers, floats, identifiers, identifiersEnd, strings, multilineComments, multilineCommentsStart, multilineCommentsEnd, webIDL;
var init_webidl = __esm({
  "../../node_modules/@codemirror/legacy-modes/mode/webidl.js"() {
    builtinArray = [
      "Clamp",
      "Constructor",
      "EnforceRange",
      "Exposed",
      "ImplicitThis",
      "Global",
      "PrimaryGlobal",
      "LegacyArrayClass",
      "LegacyUnenumerableNamedProperties",
      "LenientThis",
      "NamedConstructor",
      "NewObject",
      "NoInterfaceObject",
      "OverrideBuiltins",
      "PutForwards",
      "Replaceable",
      "SameObject",
      "TreatNonObjectAsNull",
      "TreatNullAs",
      "EmptyString",
      "Unforgeable",
      "Unscopeable"
    ];
    builtins = wordRegexp(builtinArray);
    typeArray = [
      "unsigned",
      "short",
      "long",
      // UnsignedIntegerType
      "unrestricted",
      "float",
      "double",
      // UnrestrictedFloatType
      "boolean",
      "byte",
      "octet",
      // Rest of PrimitiveType
      "Promise",
      // PromiseType
      "ArrayBuffer",
      "DataView",
      "Int8Array",
      "Int16Array",
      "Int32Array",
      "Uint8Array",
      "Uint16Array",
      "Uint32Array",
      "Uint8ClampedArray",
      "Float32Array",
      "Float64Array",
      // BufferRelatedType
      "ByteString",
      "DOMString",
      "USVString",
      "sequence",
      "object",
      "RegExp",
      "Error",
      "DOMException",
      "FrozenArray",
      // Rest of NonAnyType
      "any",
      // Rest of SingleType
      "void"
      // Rest of ReturnType
    ];
    types = wordRegexp(typeArray);
    keywordArray = [
      "attribute",
      "callback",
      "const",
      "deleter",
      "dictionary",
      "enum",
      "getter",
      "implements",
      "inherit",
      "interface",
      "iterable",
      "legacycaller",
      "maplike",
      "partial",
      "required",
      "serializer",
      "setlike",
      "setter",
      "static",
      "stringifier",
      "typedef",
      // ArgumentNameKeyword except
      // "unrestricted"
      "optional",
      "readonly",
      "or"
    ];
    keywords = wordRegexp(keywordArray);
    atomArray = [
      "true",
      "false",
      // BooleanLiteral
      "Infinity",
      "NaN",
      // FloatLiteral
      "null"
      // Rest of ConstValue
    ];
    atoms = wordRegexp(atomArray);
    startDefArray = ["callback", "dictionary", "enum", "interface"];
    startDefs = wordRegexp(startDefArray);
    endDefArray = ["typedef"];
    endDefs = wordRegexp(endDefArray);
    singleOperators = /^[:<=>?]/;
    integers = /^-?([1-9][0-9]*|0[Xx][0-9A-Fa-f]+|0[0-7]*)/;
    floats = /^-?(([0-9]+\.[0-9]*|[0-9]*\.[0-9]+)([Ee][+-]?[0-9]+)?|[0-9]+[Ee][+-]?[0-9]+)/;
    identifiers = /^_?[A-Za-z][0-9A-Z_a-z-]*/;
    identifiersEnd = /^_?[A-Za-z][0-9A-Z_a-z-]*(?=\s*;)/;
    strings = /^"[^"]*"/;
    multilineComments = /^\/\*.*?\*\//;
    multilineCommentsStart = /^\/\*.*/;
    multilineCommentsEnd = /^.*?\*\//;
    webIDL = {
      name: "webidl",
      startState: function() {
        return {
          // Is in multiline comment
          inComment: false,
          // Last non-whitespace, matched token
          lastToken: "",
          // Next token is a definition
          startDef: false,
          // Last token of the statement is a definition
          endDef: false
        };
      },
      token: function(stream, state) {
        var style = readToken(stream, state);
        if (style) {
          var cur = stream.current();
          state.lastToken = cur;
          if (style === "keyword") {
            state.startDef = startDefs.test(cur);
            state.endDef = state.endDef || endDefs.test(cur);
          } else {
            state.startDef = false;
          }
        }
        return style;
      },
      languageData: {
        autocomplete: builtinArray.concat(typeArray).concat(keywordArray).concat(atomArray)
      }
    };
  }
});
init_webidl();
export {
  webIDL
};
//# sourceMappingURL=/build/_shared/webidl-HJTPJ47D.js.map
