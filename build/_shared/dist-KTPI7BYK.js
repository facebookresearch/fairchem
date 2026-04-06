import {
  LRLanguage,
  LRParser,
  LanguageSupport,
  continuedIndent,
  foldInside,
  foldNodeProp,
  indentNodeProp,
  init_dist4 as init_dist,
  init_dist5 as init_dist2,
  init_dist6 as init_dist3,
  styleTags,
  tags
} from "/build/_shared/chunk-3I3NFLZ5.js";
import {
  __esm
} from "/build/_shared/chunk-CGOEG7L2.js";

// ../../node_modules/@lezer/json/dist/index.js
var jsonHighlighting, parser;
var init_dist4 = __esm({
  "../../node_modules/@lezer/json/dist/index.js"() {
    init_dist3();
    init_dist();
    jsonHighlighting = styleTags({
      String: tags.string,
      Number: tags.number,
      "True False": tags.bool,
      PropertyName: tags.propertyName,
      Null: tags.null,
      ", :": tags.separator,
      "[ ]": tags.squareBracket,
      "{ }": tags.brace
    });
    parser = LRParser.deserialize({
      version: 14,
      states: "$bOVQPOOOOQO'#Cb'#CbOnQPO'#CeOvQPO'#ClOOQO'#Cr'#CrQOQPOOOOQO'#Cg'#CgO}QPO'#CfO!SQPO'#CtOOQO,59P,59PO![QPO,59PO!aQPO'#CuOOQO,59W,59WO!iQPO,59WOVQPO,59QOqQPO'#CmO!nQPO,59`OOQO1G.k1G.kOVQPO'#CnO!vQPO,59aOOQO1G.r1G.rOOQO1G.l1G.lOOQO,59X,59XOOQO-E6k-E6kOOQO,59Y,59YOOQO-E6l-E6l",
      stateData: "#O~OeOS~OQSORSOSSOTSOWQO_ROgPO~OVXOgUO~O^[O~PVO[^O~O]_OVhX~OVaO~O]bO^iX~O^dO~O]_OVha~O]bO^ia~O",
      goto: "!kjPPPPPPkPPkqwPPPPk{!RPPP!XP!e!hXSOR^bQWQRf_TVQ_Q`WRg`QcZRicQTOQZRQe^RhbRYQR]R",
      nodeNames: "\u26A0 JsonText True False Null Number String } { Object Property PropertyName : , ] [ Array",
      maxTerm: 25,
      nodeProps: [
        ["isolate", -2, 6, 11, ""],
        ["openedBy", 7, "{", 14, "["],
        ["closedBy", 8, "}", 15, "]"]
      ],
      propSources: [jsonHighlighting],
      skippedNodes: [0],
      repeatNodeCount: 2,
      tokenData: "(|~RaXY!WYZ!W]^!Wpq!Wrs!]|}$u}!O$z!Q!R%T!R![&c![!]&t!}#O&y#P#Q'O#Y#Z'T#b#c'r#h#i(Z#o#p(r#q#r(w~!]Oe~~!`Wpq!]qr!]rs!xs#O!]#O#P!}#P;'S!];'S;=`$o<%lO!]~!}Og~~#QXrs!]!P!Q!]#O#P!]#U#V!]#Y#Z!]#b#c!]#f#g!]#h#i!]#i#j#m~#pR!Q![#y!c!i#y#T#Z#y~#|R!Q![$V!c!i$V#T#Z$V~$YR!Q![$c!c!i$c#T#Z$c~$fR!Q![!]!c!i!]#T#Z!]~$rP;=`<%l!]~$zO]~~$}Q!Q!R%T!R![&c~%YRT~!O!P%c!g!h%w#X#Y%w~%fP!Q![%i~%nRT~!Q![%i!g!h%w#X#Y%w~%zR{|&T}!O&T!Q![&Z~&WP!Q![&Z~&`PT~!Q![&Z~&hST~!O!P%c!Q![&c!g!h%w#X#Y%w~&yO[~~'OO_~~'TO^~~'WP#T#U'Z~'^P#`#a'a~'dP#g#h'g~'jP#X#Y'm~'rOR~~'uP#i#j'x~'{P#`#a(O~(RP#`#a(U~(ZOS~~(^P#f#g(a~(dP#i#j(g~(jP#X#Y(m~(rOQ~~(wOW~~(|OV~",
      tokenizers: [0],
      topRules: { "JsonText": [0, 1] },
      tokenPrec: 0
    });
  }
});

// ../../node_modules/@codemirror/lang-json/dist/index.js
function getErrorPosition(error, doc) {
  let m;
  if (m = error.message.match(/at position (\d+)/))
    return Math.min(+m[1], doc.length);
  if (m = error.message.match(/at line (\d+) column (\d+)/))
    return Math.min(doc.line(+m[1]).from + +m[2] - 1, doc.length);
  return 0;
}
function json() {
  return new LanguageSupport(jsonLanguage);
}
var jsonParseLinter, jsonLanguage;
var init_dist5 = __esm({
  "../../node_modules/@codemirror/lang-json/dist/index.js"() {
    init_dist4();
    init_dist2();
    jsonParseLinter = () => (view) => {
      try {
        JSON.parse(view.state.doc.toString());
      } catch (e) {
        if (!(e instanceof SyntaxError))
          throw e;
        const pos = getErrorPosition(e, view.state.doc);
        return [{
          from: pos,
          message: e.message,
          severity: "error",
          to: pos
        }];
      }
      return [];
    };
    jsonLanguage = /* @__PURE__ */ LRLanguage.define({
      name: "json",
      parser: /* @__PURE__ */ parser.configure({
        props: [
          /* @__PURE__ */ indentNodeProp.add({
            Object: /* @__PURE__ */ continuedIndent({ except: /^\s*\}/ }),
            Array: /* @__PURE__ */ continuedIndent({ except: /^\s*\]/ })
          }),
          /* @__PURE__ */ foldNodeProp.add({
            "Object Array": foldInside
          })
        ]
      }),
      languageData: {
        closeBrackets: { brackets: ["[", "{", '"'] },
        indentOnInput: /^\s*[\}\]]$/
      }
    });
  }
});
init_dist5();
export {
  json,
  jsonLanguage,
  jsonParseLinter
};
//# sourceMappingURL=/build/_shared/dist-KTPI7BYK.js.map
