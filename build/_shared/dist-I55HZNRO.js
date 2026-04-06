import {
  ContextTracker,
  EditorSelection,
  EditorView,
  ExternalTokenizer,
  LRLanguage,
  LRParser,
  LanguageSupport,
  bracketMatchingHandle,
  foldNodeProp,
  indentNodeProp,
  init_dist,
  init_dist2,
  init_dist4 as init_dist3,
  init_dist5 as init_dist4,
  init_dist6 as init_dist5,
  styleTags,
  syntaxTree,
  tags
} from "/build/_shared/chunk-3I3NFLZ5.js";
import {
  __esm
} from "/build/_shared/chunk-CGOEG7L2.js";

// ../../node_modules/@lezer/xml/dist/index.js
function nameChar(ch) {
  return ch == 45 || ch == 46 || ch == 58 || ch >= 65 && ch <= 90 || ch == 95 || ch >= 97 && ch <= 122 || ch >= 161;
}
function isSpace(ch) {
  return ch == 9 || ch == 10 || ch == 13 || ch == 32;
}
function tagNameAfter(input, offset) {
  let pos = input.pos + offset;
  if (cachedInput == input && cachedPos == pos)
    return cachedName;
  while (isSpace(input.peek(offset)))
    offset++;
  let name = "";
  for (; ; ) {
    let next = input.peek(offset);
    if (!nameChar(next))
      break;
    name += String.fromCharCode(next);
    offset++;
  }
  cachedInput = input;
  cachedPos = pos;
  return cachedName = name || null;
}
function ElementContext(name, parent) {
  this.name = name;
  this.parent = parent;
}
function scanTo(type, end) {
  return new ExternalTokenizer((input) => {
    let len = 0, first = end.charCodeAt(0);
    scan:
      for (; ; input.advance(), len++) {
        if (input.next < 0)
          break;
        if (input.next == first) {
          for (let i = 1; i < end.length; i++)
            if (input.peek(i) != end.charCodeAt(i))
              continue scan;
          break;
        }
      }
    if (len)
      input.acceptToken(type);
  });
}
var StartTag, StartCloseTag, MissingCloseTag, mismatchedStartCloseTag, incompleteStartCloseTag, commentContent$1, piContent$1, cdataContent$1, Element, OpenTag, cachedName, cachedInput, cachedPos, elementContext, startTag, commentContent, piContent, cdataContent, xmlHighlighting, parser;
var init_dist6 = __esm({
  "../../node_modules/@lezer/xml/dist/index.js"() {
    init_dist5();
    init_dist3();
    StartTag = 1;
    StartCloseTag = 2;
    MissingCloseTag = 3;
    mismatchedStartCloseTag = 4;
    incompleteStartCloseTag = 5;
    commentContent$1 = 36;
    piContent$1 = 37;
    cdataContent$1 = 38;
    Element = 11;
    OpenTag = 13;
    cachedName = null;
    cachedInput = null;
    cachedPos = 0;
    elementContext = new ContextTracker({
      start: null,
      shift(context, term, stack, input) {
        return term == StartTag ? new ElementContext(tagNameAfter(input, 1) || "", context) : context;
      },
      reduce(context, term) {
        return term == Element && context ? context.parent : context;
      },
      reuse(context, node, _stack, input) {
        let type = node.type.id;
        return type == StartTag || type == OpenTag ? new ElementContext(tagNameAfter(input, 1) || "", context) : context;
      },
      strict: false
    });
    startTag = new ExternalTokenizer((input, stack) => {
      if (input.next != 60)
        return;
      input.advance();
      if (input.next == 47) {
        input.advance();
        let name = tagNameAfter(input, 0);
        if (!name)
          return input.acceptToken(incompleteStartCloseTag);
        if (stack.context && name == stack.context.name)
          return input.acceptToken(StartCloseTag);
        for (let cx = stack.context; cx; cx = cx.parent)
          if (cx.name == name)
            return input.acceptToken(MissingCloseTag, -2);
        input.acceptToken(mismatchedStartCloseTag);
      } else if (input.next != 33 && input.next != 63) {
        return input.acceptToken(StartTag);
      }
    }, { contextual: true });
    commentContent = scanTo(commentContent$1, "-->");
    piContent = scanTo(piContent$1, "?>");
    cdataContent = scanTo(cdataContent$1, "]]>");
    xmlHighlighting = styleTags({
      Text: tags.content,
      "StartTag StartCloseTag EndTag SelfCloseEndTag": tags.angleBracket,
      TagName: tags.tagName,
      "MismatchedCloseTag/TagName": [tags.tagName, tags.invalid],
      AttributeName: tags.attributeName,
      AttributeValue: tags.attributeValue,
      Is: tags.definitionOperator,
      "EntityReference CharacterReference": tags.character,
      Comment: tags.blockComment,
      ProcessingInst: tags.processingInstruction,
      DoctypeDecl: tags.documentMeta,
      Cdata: tags.special(tags.string)
    });
    parser = LRParser.deserialize({
      version: 14,
      states: ",lOQOaOOOrOxO'#CfOzOpO'#CiO!tOaO'#CgOOOP'#Cg'#CgO!{OrO'#CrO#TOtO'#CsO#]OpO'#CtOOOP'#DT'#DTOOOP'#Cv'#CvQQOaOOOOOW'#Cw'#CwO#eOxO,59QOOOP,59Q,59QOOOO'#Cx'#CxO#mOpO,59TO#uO!bO,59TOOOP'#C|'#C|O$TOaO,59RO$[OpO'#CoOOOP,59R,59ROOOQ'#C}'#C}O$dOrO,59^OOOP,59^,59^OOOS'#DO'#DOO$lOtO,59_OOOP,59_,59_O$tOpO,59`O$|OpO,59`OOOP-E6t-E6tOOOW-E6u-E6uOOOP1G.l1G.lOOOO-E6v-E6vO%UO!bO1G.oO%UO!bO1G.oO%dOpO'#CkO%lO!bO'#CyO%zO!bO1G.oOOOP1G.o1G.oOOOP1G.w1G.wOOOP-E6z-E6zOOOP1G.m1G.mO&VOpO,59ZO&_OpO,59ZOOOQ-E6{-E6{OOOP1G.x1G.xOOOS-E6|-E6|OOOP1G.y1G.yO&gOpO1G.zO&gOpO1G.zOOOP1G.z1G.zO&oO!bO7+$ZO&}O!bO7+$ZOOOP7+$Z7+$ZOOOP7+$c7+$cO'YOpO,59VO'bOpO,59VO'mO!bO,59eOOOO-E6w-E6wO'{OpO1G.uO'{OpO1G.uOOOP1G.u1G.uO(TOpO7+$fOOOP7+$f7+$fO(]O!bO<<GuOOOP<<Gu<<GuOOOP<<G}<<G}O'bOpO1G.qO'bOpO1G.qO(hO#tO'#CnO(vO&jO'#CnOOOO1G.q1G.qO)UOpO7+$aOOOP7+$a7+$aOOOP<<HQ<<HQOOOPAN=aAN=aOOOPAN=iAN=iO'bOpO7+$]OOOO7+$]7+$]OOOO'#Cz'#CzO)^O#tO,59YOOOO,59Y,59YOOOO'#C{'#C{O)lO&jO,59YOOOP<<G{<<G{OOOO<<Gw<<GwOOOO-E6x-E6xOOOO1G.t1G.tOOOO-E6y-E6y",
      stateData: ")z~OPQOSVOTWOVWOWWOXWOiXOyPO!QTO!SUO~OvZOx]O~O^`Oz^O~OPQOQcOSVOTWOVWOWWOXWOyPO!QTO!SUO~ORdO~P!SOteO!PgO~OuhO!RjO~O^lOz^O~OvZOxoO~O^qOz^O~O[vO`sOdwOz^O~ORyO~P!SO^{Oz^O~OteO!P}O~OuhO!R!PO~O^!QOz^O~O[!SOz^O~O[!VO`sOd!WOz^O~Oa!YOz^O~Oz^O[mX`mXdmX~O[!VO`sOd!WO~O^!]Oz^O~O[!_Oz^O~O[!aOz^O~O[!cO`sOd!dOz^O~O[!cO`sOd!dO~Oa!eOz^O~Oz^O{!gO}!hO~Oz^O[ma`madma~O[!kOz^O~O[!lOz^O~O[!mO`sOd!nO~OW!qOX!qO{!sO|!qO~OW!tOX!tO}!sO!O!tO~O[!vOz^O~OW!qOX!qO{!yO|!qO~OW!tOX!tO}!yO!O!tO~O",
      goto: "%cxPPPPPPPPPPyyP!PP!VPP!`!jP!pyyyP!v!|#S$[$k$q$w$}%TPPPP%ZXWORYbXRORYb_t`qru!T!U!bQ!i!YS!p!e!fR!w!oQdRRybXSORYbQYORmYQ[PRn[Q_QQkVjp_krz!R!T!X!Z!^!`!f!j!oQr`QzcQ!RlQ!TqQ!XsQ!ZtQ!^{Q!`!QQ!f!YQ!j!]R!o!eQu`S!UqrU![u!U!bR!b!TQ!r!gR!x!rQ!u!hR!z!uQbRRxbQfTR|fQiUR!OiSXOYTaRb",
      nodeNames: "\u26A0 StartTag StartCloseTag MissingCloseTag StartCloseTag StartCloseTag Document Text EntityReference CharacterReference Cdata Element EndTag OpenTag TagName Attribute AttributeName Is AttributeValue CloseTag SelfCloseEndTag SelfClosingTag Comment ProcessingInst MismatchedCloseTag DoctypeDecl",
      maxTerm: 50,
      context: elementContext,
      nodeProps: [
        ["closedBy", 1, "SelfCloseEndTag EndTag", 13, "CloseTag MissingCloseTag"],
        ["openedBy", 12, "StartTag StartCloseTag", 19, "OpenTag", 20, "StartTag"],
        ["isolate", -6, 13, 18, 19, 21, 22, 24, ""]
      ],
      propSources: [xmlHighlighting],
      skippedNodes: [0],
      repeatNodeCount: 9,
      tokenData: "!)v~R!YOX$qXY)iYZ)iZ]$q]^)i^p$qpq)iqr$qrs*vsv$qvw+fwx/ix}$q}!O0[!O!P$q!P!Q2z!Q![$q![!]4n!]!^$q!^!_8U!_!`!#t!`!a!$l!a!b!%d!b!c$q!c!}4n!}#P$q#P#Q!'W#Q#R$q#R#S4n#S#T$q#T#o4n#o%W$q%W%o4n%o%p$q%p&a4n&a&b$q&b1p4n1p4U$q4U4d4n4d4e$q4e$IS4n$IS$I`$q$I`$Ib4n$Ib$Kh$q$Kh%#t4n%#t&/x$q&/x&Et4n&Et&FV$q&FV;'S4n;'S;:j8O;:j;=`)c<%l?&r$q?&r?Ah4n?Ah?BY$q?BY?Mn4n?MnO$qi$zXVP|W!O`Or$qrs%gsv$qwx'^x!^$q!^!_(o!_;'S$q;'S;=`)c<%lO$qa%nVVP!O`Ov%gwx&Tx!^%g!^!_&o!_;'S%g;'S;=`'W<%lO%gP&YTVPOv&Tw!^&T!_;'S&T;'S;=`&i<%lO&TP&lP;=`<%l&T`&tS!O`Ov&ox;'S&o;'S;=`'Q<%lO&o`'TP;=`<%l&oa'ZP;=`<%l%gX'eWVP|WOr'^rs&Tsv'^w!^'^!^!_'}!_;'S'^;'S;=`(i<%lO'^W(ST|WOr'}sv'}w;'S'};'S;=`(c<%lO'}W(fP;=`<%l'}X(lP;=`<%l'^h(vV|W!O`Or(ors&osv(owx'}x;'S(o;'S;=`)]<%lO(oh)`P;=`<%l(oi)fP;=`<%l$qo)t`VP|W!O`zUOX$qXY)iYZ)iZ]$q]^)i^p$qpq)iqr$qrs%gsv$qwx'^x!^$q!^!_(o!_;'S$q;'S;=`)c<%lO$qk+PV{YVP!O`Ov%gwx&Tx!^%g!^!_&o!_;'S%g;'S;=`'W<%lO%g~+iast,n![!]-r!c!}-r#R#S-r#T#o-r%W%o-r%p&a-r&b1p-r4U4d-r4e$IS-r$I`$Ib-r$Kh%#t-r&/x&Et-r&FV;'S-r;'S;:j/c?&r?Ah-r?BY?Mn-r~,qQ!Q![,w#l#m-V~,zQ!Q![,w!]!^-Q~-VOX~~-YR!Q![-c!c!i-c#T#Z-c~-fS!Q![-c!]!^-Q!c!i-c#T#Z-c~-ug}!O-r!O!P-r!Q![-r![!]-r!]!^/^!c!}-r#R#S-r#T#o-r$}%O-r%W%o-r%p&a-r&b1p-r1p4U-r4U4d-r4e$IS-r$I`$Ib-r$Je$Jg-r$Kh%#t-r&/x&Et-r&FV;'S-r;'S;:j/c?&r?Ah-r?BY?Mn-r~/cOW~~/fP;=`<%l-rk/rW}bVP|WOr'^rs&Tsv'^w!^'^!^!_'}!_;'S'^;'S;=`(i<%lO'^k0eZVP|W!O`Or$qrs%gsv$qwx'^x}$q}!O1W!O!^$q!^!_(o!_;'S$q;'S;=`)c<%lO$qk1aZVP|W!O`Or$qrs%gsv$qwx'^x!^$q!^!_(o!_!`$q!`!a2S!a;'S$q;'S;=`)c<%lO$qk2_X!PQVP|W!O`Or$qrs%gsv$qwx'^x!^$q!^!_(o!_;'S$q;'S;=`)c<%lO$qm3TZVP|W!O`Or$qrs%gsv$qwx'^x!^$q!^!_(o!_!`$q!`!a3v!a;'S$q;'S;=`)c<%lO$qm4RXdSVP|W!O`Or$qrs%gsv$qwx'^x!^$q!^!_(o!_;'S$q;'S;=`)c<%lO$qo4{!P`S^QVP|W!O`Or$qrs%gsv$qwx'^x}$q}!O4n!O!P4n!P!Q$q!Q![4n![!]4n!]!^$q!^!_(o!_!c$q!c!}4n!}#R$q#R#S4n#S#T$q#T#o4n#o$}$q$}%O4n%O%W$q%W%o4n%o%p$q%p&a4n&a&b$q&b1p4n1p4U4n4U4d4n4d4e$q4e$IS4n$IS$I`$q$I`$Ib4n$Ib$Je$q$Je$Jg4n$Jg$Kh$q$Kh%#t4n%#t&/x$q&/x&Et4n&Et&FV$q&FV;'S4n;'S;:j8O;:j;=`)c<%l?&r$q?&r?Ah4n?Ah?BY$q?BY?Mn4n?MnO$qo8RP;=`<%l4ni8]Y|W!O`Oq(oqr8{rs&osv(owx'}x!a(o!a!b!#U!b;'S(o;'S;=`)]<%lO(oi9S_|W!O`Or(ors&osv(owx'}x}(o}!O:R!O!f(o!f!g;e!g!}(o!}#ODh#O#W(o#W#XLp#X;'S(o;'S;=`)]<%lO(oi:YX|W!O`Or(ors&osv(owx'}x}(o}!O:u!O;'S(o;'S;=`)]<%lO(oi;OV!QP|W!O`Or(ors&osv(owx'}x;'S(o;'S;=`)]<%lO(oi;lX|W!O`Or(ors&osv(owx'}x!q(o!q!r<X!r;'S(o;'S;=`)]<%lO(oi<`X|W!O`Or(ors&osv(owx'}x!e(o!e!f<{!f;'S(o;'S;=`)]<%lO(oi=SX|W!O`Or(ors&osv(owx'}x!v(o!v!w=o!w;'S(o;'S;=`)]<%lO(oi=vX|W!O`Or(ors&osv(owx'}x!{(o!{!|>c!|;'S(o;'S;=`)]<%lO(oi>jX|W!O`Or(ors&osv(owx'}x!r(o!r!s?V!s;'S(o;'S;=`)]<%lO(oi?^X|W!O`Or(ors&osv(owx'}x!g(o!g!h?y!h;'S(o;'S;=`)]<%lO(oi@QY|W!O`Or?yrs@psv?yvwA[wxBdx!`?y!`!aCr!a;'S?y;'S;=`Db<%lO?ya@uV!O`Ov@pvxA[x!`@p!`!aAy!a;'S@p;'S;=`B^<%lO@pPA_TO!`A[!`!aAn!a;'SA[;'S;=`As<%lOA[PAsOiPPAvP;=`<%lA[aBQSiP!O`Ov&ox;'S&o;'S;=`'Q<%lO&oaBaP;=`<%l@pXBiX|WOrBdrsA[svBdvwA[w!`Bd!`!aCU!a;'SBd;'S;=`Cl<%lOBdXC]TiP|WOr'}sv'}w;'S'};'S;=`(c<%lO'}XCoP;=`<%lBdiC{ViP|W!O`Or(ors&osv(owx'}x;'S(o;'S;=`)]<%lO(oiDeP;=`<%l?yiDoZ|W!O`Or(ors&osv(owx'}x!e(o!e!fEb!f#V(o#V#WIr#W;'S(o;'S;=`)]<%lO(oiEiX|W!O`Or(ors&osv(owx'}x!f(o!f!gFU!g;'S(o;'S;=`)]<%lO(oiF]X|W!O`Or(ors&osv(owx'}x!c(o!c!dFx!d;'S(o;'S;=`)]<%lO(oiGPX|W!O`Or(ors&osv(owx'}x!v(o!v!wGl!w;'S(o;'S;=`)]<%lO(oiGsX|W!O`Or(ors&osv(owx'}x!c(o!c!dH`!d;'S(o;'S;=`)]<%lO(oiHgX|W!O`Or(ors&osv(owx'}x!}(o!}#OIS#O;'S(o;'S;=`)]<%lO(oiI]V|W!O`yPOr(ors&osv(owx'}x;'S(o;'S;=`)]<%lO(oiIyX|W!O`Or(ors&osv(owx'}x#W(o#W#XJf#X;'S(o;'S;=`)]<%lO(oiJmX|W!O`Or(ors&osv(owx'}x#T(o#T#UKY#U;'S(o;'S;=`)]<%lO(oiKaX|W!O`Or(ors&osv(owx'}x#h(o#h#iK|#i;'S(o;'S;=`)]<%lO(oiLTX|W!O`Or(ors&osv(owx'}x#T(o#T#UH`#U;'S(o;'S;=`)]<%lO(oiLwX|W!O`Or(ors&osv(owx'}x#c(o#c#dMd#d;'S(o;'S;=`)]<%lO(oiMkX|W!O`Or(ors&osv(owx'}x#V(o#V#WNW#W;'S(o;'S;=`)]<%lO(oiN_X|W!O`Or(ors&osv(owx'}x#h(o#h#iNz#i;'S(o;'S;=`)]<%lO(oi! RX|W!O`Or(ors&osv(owx'}x#m(o#m#n! n#n;'S(o;'S;=`)]<%lO(oi! uX|W!O`Or(ors&osv(owx'}x#d(o#d#e!!b#e;'S(o;'S;=`)]<%lO(oi!!iX|W!O`Or(ors&osv(owx'}x#X(o#X#Y?y#Y;'S(o;'S;=`)]<%lO(oi!#_V!SP|W!O`Or(ors&osv(owx'}x;'S(o;'S;=`)]<%lO(ok!$PXaQVP|W!O`Or$qrs%gsv$qwx'^x!^$q!^!_(o!_;'S$q;'S;=`)c<%lO$qo!$wX[UVP|W!O`Or$qrs%gsv$qwx'^x!^$q!^!_(o!_;'S$q;'S;=`)c<%lO$qk!%mZVP|W!O`Or$qrs%gsv$qwx'^x!^$q!^!_(o!_!`$q!`!a!&`!a;'S$q;'S;=`)c<%lO$qk!&kX!RQVP|W!O`Or$qrs%gsv$qwx'^x!^$q!^!_(o!_;'S$q;'S;=`)c<%lO$qk!'aZVP|W!O`Or$qrs%gsv$qwx'^x!^$q!^!_(o!_#P$q#P#Q!(S#Q;'S$q;'S;=`)c<%lO$qk!(]ZVP|W!O`Or$qrs%gsv$qwx'^x!^$q!^!_(o!_!`$q!`!a!)O!a;'S$q;'S;=`)c<%lO$qk!)ZXxQVP|W!O`Or$qrs%gsv$qwx'^x!^$q!^!_(o!_;'S$q;'S;=`)c<%lO$q",
      tokenizers: [startTag, commentContent, piContent, cdataContent, 0, 1, 2, 3, 4],
      topRules: { "Document": [0, 6] },
      tokenPrec: 0
    });
  }
});

// ../../node_modules/@codemirror/lang-xml/dist/index.js
function tagName(doc, tag) {
  let name = tag && tag.getChild("TagName");
  return name ? doc.sliceString(name.from, name.to) : "";
}
function elementName$1(doc, tree) {
  let tag = tree && tree.firstChild;
  return !tag || tag.name != "OpenTag" ? "" : tagName(doc, tag);
}
function attrName(doc, tag, pos) {
  let attr = tag && tag.getChildren("Attribute").find((a) => a.from <= pos && a.to >= pos);
  let name = attr && attr.getChild("AttributeName");
  return name ? doc.sliceString(name.from, name.to) : "";
}
function findParentElement(tree) {
  for (let cur = tree && tree.parent; cur; cur = cur.parent)
    if (cur.name == "Element")
      return cur;
  return null;
}
function findLocation(state, pos) {
  var _a;
  let at = syntaxTree(state).resolveInner(pos, -1), inTag = null;
  for (let cur = at; !inTag && cur.parent; cur = cur.parent)
    if (cur.name == "OpenTag" || cur.name == "CloseTag" || cur.name == "SelfClosingTag" || cur.name == "MismatchedCloseTag")
      inTag = cur;
  if (inTag && (inTag.to > pos || inTag.lastChild.type.isError)) {
    let elt = inTag.parent;
    if (at.name == "TagName")
      return inTag.name == "CloseTag" || inTag.name == "MismatchedCloseTag" ? { type: "closeTag", from: at.from, context: elt } : { type: "openTag", from: at.from, context: findParentElement(elt) };
    if (at.name == "AttributeName")
      return { type: "attrName", from: at.from, context: inTag };
    if (at.name == "AttributeValue")
      return { type: "attrValue", from: at.from, context: inTag };
    let before = at == inTag || at.name == "Attribute" ? at.childBefore(pos) : at;
    if ((before === null || before === void 0 ? void 0 : before.name) == "StartTag")
      return { type: "openTag", from: pos, context: findParentElement(elt) };
    if ((before === null || before === void 0 ? void 0 : before.name) == "StartCloseTag" && before.to <= pos)
      return { type: "closeTag", from: pos, context: elt };
    if ((before === null || before === void 0 ? void 0 : before.name) == "Is")
      return { type: "attrValue", from: pos, context: inTag };
    if (before)
      return { type: "attrName", from: pos, context: inTag };
    return null;
  } else if (at.name == "StartCloseTag") {
    return { type: "closeTag", from: pos, context: at.parent };
  }
  while (at.parent && at.to == pos && !((_a = at.lastChild) === null || _a === void 0 ? void 0 : _a.type.isError))
    at = at.parent;
  if (at.name == "Element" || at.name == "Text" || at.name == "Document")
    return { type: "tag", from: pos, context: at.name == "Element" ? at : findParentElement(at) };
  return null;
}
function attrCompletion(spec) {
  return Object.assign(Object.assign({ type: "property" }, spec.completion || {}), { label: spec.name });
}
function valueCompletion(spec) {
  return typeof spec == "string" ? { label: `"${spec}"`, type: "constant" } : /^"/.test(spec.label) ? spec : Object.assign(Object.assign({}, spec), { label: `"${spec.label}"` });
}
function completeFromSchema(eltSpecs, attrSpecs) {
  let allAttrs = [], globalAttrs = [];
  let attrValues = /* @__PURE__ */ Object.create(null);
  for (let s of attrSpecs) {
    let completion = attrCompletion(s);
    allAttrs.push(completion);
    if (s.global)
      globalAttrs.push(completion);
    if (s.values)
      attrValues[s.name] = s.values.map(valueCompletion);
  }
  let allElements = [], topElements = [];
  let byName = /* @__PURE__ */ Object.create(null);
  for (let s of eltSpecs) {
    let attrs = globalAttrs, attrVals = attrValues;
    if (s.attributes)
      attrs = attrs.concat(s.attributes.map((s2) => {
        if (typeof s2 == "string")
          return allAttrs.find((a) => a.label == s2) || { label: s2, type: "property" };
        if (s2.values) {
          if (attrVals == attrValues)
            attrVals = Object.create(attrVals);
          attrVals[s2.name] = s2.values.map(valueCompletion);
        }
        return attrCompletion(s2);
      }));
    let elt = new Element2(s, attrs, attrVals);
    byName[elt.name] = elt;
    allElements.push(elt);
    if (s.top)
      topElements.push(elt);
  }
  if (!topElements.length)
    topElements = allElements;
  for (let i = 0; i < allElements.length; i++) {
    let s = eltSpecs[i], elt = allElements[i];
    if (s.children) {
      for (let ch of s.children)
        if (byName[ch])
          elt.children.push(byName[ch]);
    } else {
      elt.children = allElements;
    }
  }
  return (cx) => {
    var _a;
    let { doc } = cx.state, loc = findLocation(cx.state, cx.pos);
    if (!loc || loc.type == "tag" && !cx.explicit)
      return null;
    let { type, from, context } = loc;
    if (type == "openTag") {
      let children = topElements;
      let parentName = elementName$1(doc, context);
      if (parentName) {
        let parent = byName[parentName];
        children = (parent === null || parent === void 0 ? void 0 : parent.children) || allElements;
      }
      return {
        from,
        options: children.map((ch) => ch.completion),
        validFor: Identifier
      };
    } else if (type == "closeTag") {
      let parentName = elementName$1(doc, context);
      return parentName ? {
        from,
        to: cx.pos + (doc.sliceString(cx.pos, cx.pos + 1) == ">" ? 1 : 0),
        options: [((_a = byName[parentName]) === null || _a === void 0 ? void 0 : _a.closeNameCompletion) || { label: parentName + ">", type: "type" }],
        validFor: Identifier
      } : null;
    } else if (type == "attrName") {
      let parent = byName[tagName(doc, context)];
      return {
        from,
        options: (parent === null || parent === void 0 ? void 0 : parent.attrs) || globalAttrs,
        validFor: Identifier
      };
    } else if (type == "attrValue") {
      let attr = attrName(doc, context, from);
      if (!attr)
        return null;
      let parent = byName[tagName(doc, context)];
      let values = ((parent === null || parent === void 0 ? void 0 : parent.attrValues) || attrValues)[attr];
      if (!values || !values.length)
        return null;
      return {
        from,
        to: cx.pos + (doc.sliceString(cx.pos, cx.pos + 1) == '"' ? 1 : 0),
        options: values,
        validFor: /^"[^"]*"?$/
      };
    } else if (type == "tag") {
      let parentName = elementName$1(doc, context), parent = byName[parentName];
      let closing = [], last = context && context.lastChild;
      if (parentName && (!last || last.name != "CloseTag" || tagName(doc, last) != parentName))
        closing.push(parent ? parent.closeCompletion : { label: "</" + parentName + ">", type: "type", boost: 2 });
      let options = closing.concat(((parent === null || parent === void 0 ? void 0 : parent.children) || (context ? allElements : topElements)).map((e) => e.openCompletion));
      if (context && (parent === null || parent === void 0 ? void 0 : parent.text.length)) {
        let openTag = context.firstChild;
        if (openTag.to > cx.pos - 20 && !/\S/.test(cx.state.sliceDoc(openTag.to, cx.pos)))
          options = options.concat(parent.text);
      }
      return {
        from,
        options,
        validFor: /^<\/?[:\-\.\w\u00b7-\uffff]*$/
      };
    } else {
      return null;
    }
  };
}
function xml(conf = {}) {
  let support = [xmlLanguage.data.of({
    autocomplete: completeFromSchema(conf.elements || [], conf.attributes || [])
  })];
  if (conf.autoCloseTags !== false)
    support.push(autoCloseTags);
  return new LanguageSupport(xmlLanguage, support);
}
function elementName(doc, tree, max = doc.length) {
  if (!tree)
    return "";
  let tag = tree.firstChild;
  let name = tag && tag.getChild("TagName");
  return name ? doc.sliceString(name.from, Math.min(name.to, max)) : "";
}
var Element2, Identifier, xmlLanguage, autoCloseTags;
var init_dist7 = __esm({
  "../../node_modules/@codemirror/lang-xml/dist/index.js"() {
    init_dist6();
    init_dist4();
    init_dist();
    init_dist2();
    Element2 = class {
      constructor(spec, attrs, attrValues) {
        this.attrs = attrs;
        this.attrValues = attrValues;
        this.children = [];
        this.name = spec.name;
        this.completion = Object.assign(Object.assign({ type: "type" }, spec.completion || {}), { label: this.name });
        this.openCompletion = Object.assign(Object.assign({}, this.completion), { label: "<" + this.name });
        this.closeCompletion = Object.assign(Object.assign({}, this.completion), { label: "</" + this.name + ">", boost: 2 });
        this.closeNameCompletion = Object.assign(Object.assign({}, this.completion), { label: this.name + ">" });
        this.text = spec.textContent ? spec.textContent.map((s) => ({ label: s, type: "text" })) : [];
      }
    };
    Identifier = /^[:\-\.\w\u00b7-\uffff]*$/;
    xmlLanguage = /* @__PURE__ */ LRLanguage.define({
      name: "xml",
      parser: /* @__PURE__ */ parser.configure({
        props: [
          /* @__PURE__ */ indentNodeProp.add({
            Element(context) {
              let closed = /^\s*<\//.test(context.textAfter);
              return context.lineIndent(context.node.from) + (closed ? 0 : context.unit);
            },
            "OpenTag CloseTag SelfClosingTag"(context) {
              return context.column(context.node.from) + context.unit;
            }
          }),
          /* @__PURE__ */ foldNodeProp.add({
            Element(subtree) {
              let first = subtree.firstChild, last = subtree.lastChild;
              if (!first || first.name != "OpenTag")
                return null;
              return { from: first.to, to: last.name == "CloseTag" ? last.from : subtree.to };
            }
          }),
          /* @__PURE__ */ bracketMatchingHandle.add({
            "OpenTag CloseTag": (node) => node.getChild("TagName")
          })
        ]
      }),
      languageData: {
        commentTokens: { block: { open: "<!--", close: "-->" } },
        indentOnInput: /^\s*<\/$/
      }
    });
    autoCloseTags = /* @__PURE__ */ EditorView.inputHandler.of((view, from, to, text, insertTransaction) => {
      if (view.composing || view.state.readOnly || from != to || text != ">" && text != "/" || !xmlLanguage.isActiveAt(view.state, from, -1))
        return false;
      let base = insertTransaction(), { state } = base;
      let closeTags = state.changeByRange((range) => {
        var _a, _b, _c;
        let { head } = range;
        let didType = state.doc.sliceString(head - 1, head) == text;
        let after = syntaxTree(state).resolveInner(head, -1), name;
        if (didType && text == ">" && after.name == "EndTag") {
          let tag = after.parent;
          if (((_b = (_a = tag.parent) === null || _a === void 0 ? void 0 : _a.lastChild) === null || _b === void 0 ? void 0 : _b.name) != "CloseTag" && (name = elementName(state.doc, tag.parent, head))) {
            let to2 = head + (state.doc.sliceString(head, head + 1) === ">" ? 1 : 0);
            let insert = `</${name}>`;
            return { range, changes: { from: head, to: to2, insert } };
          }
        } else if (didType && text == "/" && after.name == "StartCloseTag") {
          let base2 = after.parent;
          if (after.from == head - 2 && ((_c = base2.lastChild) === null || _c === void 0 ? void 0 : _c.name) != "CloseTag" && (name = elementName(state.doc, base2, head))) {
            let to2 = head + (state.doc.sliceString(head, head + 1) === ">" ? 1 : 0);
            let insert = `${name}>`;
            return {
              range: EditorSelection.cursor(head + insert.length, -1),
              changes: { from: head, to: to2, insert }
            };
          }
        }
        return { range };
      });
      if (closeTags.changes.empty)
        return false;
      view.dispatch([
        base,
        state.update(closeTags, {
          userEvent: "input.complete",
          scrollIntoView: true
        })
      ]);
      return true;
    });
  }
});
init_dist7();
export {
  autoCloseTags,
  completeFromSchema,
  xml,
  xmlLanguage
};
//# sourceMappingURL=/build/_shared/dist-I55HZNRO.js.map
