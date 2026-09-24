import{a as Vt}from"/fairchem/build/_shared/chunk-D6F2AK33.js";import{a as Mt}from"/fairchem/build/_shared/chunk-4BXGAWUT.js";import{b as Yt}from"/fairchem/build/_shared/chunk-AS4OQ6CY.js";import{g as Bt,p as Gt}from"/fairchem/build/_shared/chunk-II7SFVQZ.js";import{N as V,T as Ot,U as Nt,V as Rt,W as wt,X as $t,Y as Pt,Z as Ft,_ as w}from"/fairchem/build/_shared/chunk-ZHZ3OZD7.js";import{b as u,d as E}from"/fairchem/build/_shared/chunk-DYFJVR7X.js";var vt=function(){var t=u(function(Y,a,c,i){for(c=c||{},i=Y.length;i--;c[Y[i]]=a);return c},"o"),e=[1,2],l=[1,3],s=[1,4],d=[2,4],f=[1,9],S=[1,11],g=[1,16],o=[1,17],T=[1,18],k=[1,19],N=[1,33],A=[1,20],m=[1,21],h=[1,22],x=[1,23],D=[1,24],$=[1,26],L=[1,27],P=[1,28],I=[1,29],J=[1,30],st=[1,31],rt=[1,32],it=[1,35],at=[1,36],nt=[1,37],ot=[1,38],j=[1,34],p=[1,4,5,16,17,19,21,22,24,25,26,27,28,29,33,35,37,38,41,45,48,51,52,53,54,57],lt=[1,4,5,14,15,16,17,19,21,22,24,25,26,27,28,29,33,35,37,38,39,40,41,45,48,51,52,53,54,57],xt=[4,5,16,17,19,21,22,24,25,26,27,28,29,33,35,37,38,41,45,48,51,52,53,54,57],gt={trace:u(function(){},"trace"),yy:{},symbols_:{error:2,start:3,SPACE:4,NL:5,SD:6,document:7,line:8,statement:9,classDefStatement:10,styleStatement:11,cssClassStatement:12,idStatement:13,DESCR:14,"-->":15,HIDE_EMPTY:16,scale:17,WIDTH:18,COMPOSIT_STATE:19,STRUCT_START:20,STRUCT_STOP:21,STATE_DESCR:22,AS:23,ID:24,FORK:25,JOIN:26,CHOICE:27,CONCURRENT:28,note:29,notePosition:30,NOTE_TEXT:31,direction:32,acc_title:33,acc_title_value:34,acc_descr:35,acc_descr_value:36,acc_descr_multiline_value:37,CLICK:38,STRING:39,HREF:40,classDef:41,CLASSDEF_ID:42,CLASSDEF_STYLEOPTS:43,DEFAULT:44,style:45,STYLE_IDS:46,STYLEDEF_STYLEOPTS:47,class:48,CLASSENTITY_IDS:49,STYLECLASS:50,direction_tb:51,direction_bt:52,direction_rl:53,direction_lr:54,eol:55,";":56,EDGE_STATE:57,STYLE_SEPARATOR:58,left_of:59,right_of:60,$accept:0,$end:1},terminals_:{2:"error",4:"SPACE",5:"NL",6:"SD",14:"DESCR",15:"-->",16:"HIDE_EMPTY",17:"scale",18:"WIDTH",19:"COMPOSIT_STATE",20:"STRUCT_START",21:"STRUCT_STOP",22:"STATE_DESCR",23:"AS",24:"ID",25:"FORK",26:"JOIN",27:"CHOICE",28:"CONCURRENT",29:"note",31:"NOTE_TEXT",33:"acc_title",34:"acc_title_value",35:"acc_descr",36:"acc_descr_value",37:"acc_descr_multiline_value",38:"CLICK",39:"STRING",40:"HREF",41:"classDef",42:"CLASSDEF_ID",43:"CLASSDEF_STYLEOPTS",44:"DEFAULT",45:"style",46:"STYLE_IDS",47:"STYLEDEF_STYLEOPTS",48:"class",49:"CLASSENTITY_IDS",50:"STYLECLASS",51:"direction_tb",52:"direction_bt",53:"direction_rl",54:"direction_lr",56:";",57:"EDGE_STATE",58:"STYLE_SEPARATOR",59:"left_of",60:"right_of"},productions_:[0,[3,2],[3,2],[3,2],[7,0],[7,2],[8,2],[8,1],[8,1],[9,1],[9,1],[9,1],[9,1],[9,2],[9,3],[9,4],[9,1],[9,2],[9,1],[9,4],[9,3],[9,6],[9,1],[9,1],[9,1],[9,1],[9,4],[9,4],[9,1],[9,2],[9,2],[9,1],[9,5],[9,5],[10,3],[10,3],[11,3],[12,3],[32,1],[32,1],[32,1],[32,1],[55,1],[55,1],[13,1],[13,1],[13,3],[13,3],[30,1],[30,1]],performAction:u(function(a,c,i,y,b,r,H){var n=r.length-1;switch(b){case 3:return y.setRootDoc(r[n]),r[n];break;case 4:this.$=[];break;case 5:r[n]!="nl"&&(r[n-1].push(r[n]),this.$=r[n-1]);break;case 6:case 7:this.$=r[n];break;case 8:this.$="nl";break;case 12:this.$=r[n];break;case 13:let ht=r[n-1];ht.description=y.trimColon(r[n]),this.$=ht;break;case 14:this.$={stmt:"relation",state1:r[n-2],state2:r[n]};break;case 15:let ut=y.trimColon(r[n]);this.$={stmt:"relation",state1:r[n-3],state2:r[n-1],description:ut};break;case 19:this.$={stmt:"state",id:r[n-3],type:"default",description:"",doc:r[n-1]};break;case 20:var F=r[n],B=r[n-2].trim();if(r[n].match(":")){var q=r[n].split(":");F=q[0],B=[B,q[1]]}this.$={stmt:"state",id:F,type:"default",description:B};break;case 21:this.$={stmt:"state",id:r[n-3],type:"default",description:r[n-5],doc:r[n-1]};break;case 22:this.$={stmt:"state",id:r[n],type:"fork"};break;case 23:this.$={stmt:"state",id:r[n],type:"join"};break;case 24:this.$={stmt:"state",id:r[n],type:"choice"};break;case 25:this.$={stmt:"state",id:y.getDividerId(),type:"divider"};break;case 26:this.$={stmt:"state",id:r[n-1].trim(),note:{position:r[n-2].trim(),text:r[n].trim()}};break;case 29:this.$=r[n].trim(),y.setAccTitle(this.$);break;case 30:case 31:this.$=r[n].trim(),y.setAccDescription(this.$);break;case 32:this.$={stmt:"click",id:r[n-3],url:r[n-2],tooltip:r[n-1]};break;case 33:this.$={stmt:"click",id:r[n-3],url:r[n-1],tooltip:""};break;case 34:case 35:this.$={stmt:"classDef",id:r[n-1].trim(),classes:r[n].trim()};break;case 36:this.$={stmt:"style",id:r[n-1].trim(),styleClass:r[n].trim()};break;case 37:this.$={stmt:"applyClass",id:r[n-1].trim(),styleClass:r[n].trim()};break;case 38:y.setDirection("TB"),this.$={stmt:"dir",value:"TB"};break;case 39:y.setDirection("BT"),this.$={stmt:"dir",value:"BT"};break;case 40:y.setDirection("RL"),this.$={stmt:"dir",value:"RL"};break;case 41:y.setDirection("LR"),this.$={stmt:"dir",value:"LR"};break;case 44:case 45:this.$={stmt:"state",id:r[n].trim(),type:"default",description:""};break;case 46:this.$={stmt:"state",id:r[n-2].trim(),classes:[r[n].trim()],type:"default",description:""};break;case 47:this.$={stmt:"state",id:r[n-2].trim(),classes:[r[n].trim()],type:"default",description:""};break}},"anonymous"),table:[{3:1,4:e,5:l,6:s},{1:[3]},{3:5,4:e,5:l,6:s},{3:6,4:e,5:l,6:s},t([1,4,5,16,17,19,22,24,25,26,27,28,29,33,35,37,38,41,45,48,51,52,53,54,57],d,{7:7}),{1:[2,1]},{1:[2,2]},{1:[2,3],4:f,5:S,8:8,9:10,10:12,11:13,12:14,13:15,16:g,17:o,19:T,22:k,24:N,25:A,26:m,27:h,28:x,29:D,32:25,33:$,35:L,37:P,38:I,41:J,45:st,48:rt,51:it,52:at,53:nt,54:ot,57:j},t(p,[2,5]),{9:39,10:12,11:13,12:14,13:15,16:g,17:o,19:T,22:k,24:N,25:A,26:m,27:h,28:x,29:D,32:25,33:$,35:L,37:P,38:I,41:J,45:st,48:rt,51:it,52:at,53:nt,54:ot,57:j},t(p,[2,7]),t(p,[2,8]),t(p,[2,9]),t(p,[2,10]),t(p,[2,11]),t(p,[2,12],{14:[1,40],15:[1,41]}),t(p,[2,16]),{18:[1,42]},t(p,[2,18],{20:[1,43]}),{23:[1,44]},t(p,[2,22]),t(p,[2,23]),t(p,[2,24]),t(p,[2,25]),{30:45,31:[1,46],59:[1,47],60:[1,48]},t(p,[2,28]),{34:[1,49]},{36:[1,50]},t(p,[2,31]),{13:51,24:N,57:j},{42:[1,52],44:[1,53]},{46:[1,54]},{49:[1,55]},t(lt,[2,44],{58:[1,56]}),t(lt,[2,45],{58:[1,57]}),t(p,[2,38]),t(p,[2,39]),t(p,[2,40]),t(p,[2,41]),t(p,[2,6]),t(p,[2,13]),{13:58,24:N,57:j},t(p,[2,17]),t(xt,d,{7:59}),{24:[1,60]},{24:[1,61]},{23:[1,62]},{24:[2,48]},{24:[2,49]},t(p,[2,29]),t(p,[2,30]),{39:[1,63],40:[1,64]},{43:[1,65]},{43:[1,66]},{47:[1,67]},{50:[1,68]},{24:[1,69]},{24:[1,70]},t(p,[2,14],{14:[1,71]}),{4:f,5:S,8:8,9:10,10:12,11:13,12:14,13:15,16:g,17:o,19:T,21:[1,72],22:k,24:N,25:A,26:m,27:h,28:x,29:D,32:25,33:$,35:L,37:P,38:I,41:J,45:st,48:rt,51:it,52:at,53:nt,54:ot,57:j},t(p,[2,20],{20:[1,73]}),{31:[1,74]},{24:[1,75]},{39:[1,76]},{39:[1,77]},t(p,[2,34]),t(p,[2,35]),t(p,[2,36]),t(p,[2,37]),t(lt,[2,46]),t(lt,[2,47]),t(p,[2,15]),t(p,[2,19]),t(xt,d,{7:78}),t(p,[2,26]),t(p,[2,27]),{5:[1,79]},{5:[1,80]},{4:f,5:S,8:8,9:10,10:12,11:13,12:14,13:15,16:g,17:o,19:T,21:[1,81],22:k,24:N,25:A,26:m,27:h,28:x,29:D,32:25,33:$,35:L,37:P,38:I,41:J,45:st,48:rt,51:it,52:at,53:nt,54:ot,57:j},t(p,[2,32]),t(p,[2,33]),t(p,[2,21])],defaultActions:{5:[2,1],6:[2,2],47:[2,48],48:[2,49]},parseError:u(function(a,c){if(c.recoverable)this.trace(a);else{var i=new Error(a);throw i.hash=c,i}},"parseError"),parse:u(function(a){var c=this,i=[0],y=[],b=[null],r=[],H=this.table,n="",F=0,B=0,q=0,ht=2,ut=1,he=r.slice.call(arguments,1),_=Object.create(this.lexer),U={yy:{}};for(var Tt in this.yy)Object.prototype.hasOwnProperty.call(this.yy,Tt)&&(U.yy[Tt]=this.yy[Tt]);_.setInput(a,U.yy),U.yy.lexer=_,U.yy.parser=this,typeof _.yylloc>"u"&&(_.yylloc={});var bt=_.yylloc;r.push(bt);var ue=_.options&&_.options.ranges;typeof U.yy.parseError=="function"?this.parseError=U.yy.parseError:this.parseError=Object.getPrototypeOf(this).parseError;function de(O){i.length=i.length-2*O,b.length=b.length-O,r.length=r.length-O}u(de,"popStack");function Lt(){var O;return O=y.pop()||_.lex()||ut,typeof O!="number"&&(O instanceof Array&&(y=O,O=y.pop()),O=c.symbols_[O]||O),O}u(Lt,"lex");for(var v,Et,W,R,Be,kt,z={},dt,G,It,ft;;){if(W=i[i.length-1],this.defaultActions[W]?R=this.defaultActions[W]:((v===null||typeof v>"u")&&(v=Lt()),R=H[W]&&H[W][v]),typeof R>"u"||!R.length||!R[0]){var _t="";ft=[];for(dt in H[W])this.terminals_[dt]&&dt>ht&&ft.push("'"+this.terminals_[dt]+"'");_.showPosition?_t="Parse error on line "+(F+1)+`:
`+_.showPosition()+`
Expecting `+ft.join(", ")+", got '"+(this.terminals_[v]||v)+"'":_t="Parse error on line "+(F+1)+": Unexpected "+(v==ut?"end of input":"'"+(this.terminals_[v]||v)+"'"),this.parseError(_t,{text:_.match,token:this.terminals_[v]||v,line:_.yylineno,loc:bt,expected:ft})}if(R[0]instanceof Array&&R.length>1)throw new Error("Parse Error: multiple actions possible at state: "+W+", token: "+v);switch(R[0]){case 1:i.push(v),b.push(_.yytext),r.push(_.yylloc),i.push(R[1]),v=null,Et?(v=Et,Et=null):(B=_.yyleng,n=_.yytext,F=_.yylineno,bt=_.yylloc,q>0&&q--);break;case 2:if(G=this.productions_[R[1]][1],z.$=b[b.length-G],z._$={first_line:r[r.length-(G||1)].first_line,last_line:r[r.length-1].last_line,first_column:r[r.length-(G||1)].first_column,last_column:r[r.length-1].last_column},ue&&(z._$.range=[r[r.length-(G||1)].range[0],r[r.length-1].range[1]]),kt=this.performAction.apply(z,[n,B,F,U.yy,R[1],b,r].concat(he)),typeof kt<"u")return kt;G&&(i=i.slice(0,-1*G*2),b=b.slice(0,-1*G),r=r.slice(0,-1*G)),i.push(this.productions_[R[1]][0]),b.push(z.$),r.push(z._$),It=H[i[i.length-2]][i[i.length-1]],i.push(It);break;case 3:return!0}}return!0},"parse")},ce=function(){var Y={EOF:1,parseError:u(function(c,i){if(this.yy.parser)this.yy.parser.parseError(c,i);else throw new Error(c)},"parseError"),setInput:u(function(a,c){return this.yy=c||this.yy||{},this._input=a,this._more=this._backtrack=this.done=!1,this.yylineno=this.yyleng=0,this.yytext=this.matched=this.match="",this.conditionStack=["INITIAL"],this.yylloc={first_line:1,first_column:0,last_line:1,last_column:0},this.options.ranges&&(this.yylloc.range=[0,0]),this.offset=0,this},"setInput"),input:u(function(){var a=this._input[0];this.yytext+=a,this.yyleng++,this.offset++,this.match+=a,this.matched+=a;var c=a.match(/(?:\r\n?|\n).*/g);return c?(this.yylineno++,this.yylloc.last_line++):this.yylloc.last_column++,this.options.ranges&&this.yylloc.range[1]++,this._input=this._input.slice(1),a},"input"),unput:u(function(a){var c=a.length,i=a.split(/(?:\r\n?|\n)/g);this._input=a+this._input,this.yytext=this.yytext.substr(0,this.yytext.length-c),this.offset-=c;var y=this.match.split(/(?:\r\n?|\n)/g);this.match=this.match.substr(0,this.match.length-1),this.matched=this.matched.substr(0,this.matched.length-1),i.length-1&&(this.yylineno-=i.length-1);var b=this.yylloc.range;return this.yylloc={first_line:this.yylloc.first_line,last_line:this.yylineno+1,first_column:this.yylloc.first_column,last_column:i?(i.length===y.length?this.yylloc.first_column:0)+y[y.length-i.length].length-i[0].length:this.yylloc.first_column-c},this.options.ranges&&(this.yylloc.range=[b[0],b[0]+this.yyleng-c]),this.yyleng=this.yytext.length,this},"unput"),more:u(function(){return this._more=!0,this},"more"),reject:u(function(){if(this.options.backtrack_lexer)this._backtrack=!0;else return this.parseError("Lexical error on line "+(this.yylineno+1)+`. You can only invoke reject() in the lexer when the lexer is of the backtracking persuasion (options.backtrack_lexer = true).
`+this.showPosition(),{text:"",token:null,line:this.yylineno});return this},"reject"),less:u(function(a){this.unput(this.match.slice(a))},"less"),pastInput:u(function(){var a=this.matched.substr(0,this.matched.length-this.match.length);return(a.length>20?"...":"")+a.substr(-20).replace(/\n/g,"")},"pastInput"),upcomingInput:u(function(){var a=this.match;return a.length<20&&(a+=this._input.substr(0,20-a.length)),(a.substr(0,20)+(a.length>20?"...":"")).replace(/\n/g,"")},"upcomingInput"),showPosition:u(function(){var a=this.pastInput(),c=new Array(a.length+1).join("-");return a+this.upcomingInput()+`
`+c+"^"},"showPosition"),test_match:u(function(a,c){var i,y,b;if(this.options.backtrack_lexer&&(b={yylineno:this.yylineno,yylloc:{first_line:this.yylloc.first_line,last_line:this.last_line,first_column:this.yylloc.first_column,last_column:this.yylloc.last_column},yytext:this.yytext,match:this.match,matches:this.matches,matched:this.matched,yyleng:this.yyleng,offset:this.offset,_more:this._more,_input:this._input,yy:this.yy,conditionStack:this.conditionStack.slice(0),done:this.done},this.options.ranges&&(b.yylloc.range=this.yylloc.range.slice(0))),y=a[0].match(/(?:\r\n?|\n).*/g),y&&(this.yylineno+=y.length),this.yylloc={first_line:this.yylloc.last_line,last_line:this.yylineno+1,first_column:this.yylloc.last_column,last_column:y?y[y.length-1].length-y[y.length-1].match(/\r?\n?/)[0].length:this.yylloc.last_column+a[0].length},this.yytext+=a[0],this.match+=a[0],this.matches=a,this.yyleng=this.yytext.length,this.options.ranges&&(this.yylloc.range=[this.offset,this.offset+=this.yyleng]),this._more=!1,this._backtrack=!1,this._input=this._input.slice(a[0].length),this.matched+=a[0],i=this.performAction.call(this,this.yy,this,c,this.conditionStack[this.conditionStack.length-1]),this.done&&this._input&&(this.done=!1),i)return i;if(this._backtrack){for(var r in b)this[r]=b[r];return!1}return!1},"test_match"),next:u(function(){if(this.done)return this.EOF;this._input||(this.done=!0);var a,c,i,y;this._more||(this.yytext="",this.match="");for(var b=this._currentRules(),r=0;r<b.length;r++)if(i=this._input.match(this.rules[b[r]]),i&&(!c||i[0].length>c[0].length)){if(c=i,y=r,this.options.backtrack_lexer){if(a=this.test_match(i,b[r]),a!==!1)return a;if(this._backtrack){c=!1;continue}else return!1}else if(!this.options.flex)break}return c?(a=this.test_match(c,b[y]),a!==!1?a:!1):this._input===""?this.EOF:this.parseError("Lexical error on line "+(this.yylineno+1)+`. Unrecognized text.
`+this.showPosition(),{text:"",token:null,line:this.yylineno})},"next"),lex:u(function(){var c=this.next();return c||this.lex()},"lex"),begin:u(function(c){this.conditionStack.push(c)},"begin"),popState:u(function(){var c=this.conditionStack.length-1;return c>0?this.conditionStack.pop():this.conditionStack[0]},"popState"),_currentRules:u(function(){return this.conditionStack.length&&this.conditionStack[this.conditionStack.length-1]?this.conditions[this.conditionStack[this.conditionStack.length-1]].rules:this.conditions.INITIAL.rules},"_currentRules"),topState:u(function(c){return c=this.conditionStack.length-1-Math.abs(c||0),c>=0?this.conditionStack[c]:"INITIAL"},"topState"),pushState:u(function(c){this.begin(c)},"pushState"),stateStackSize:u(function(){return this.conditionStack.length},"stateStackSize"),options:{"case-insensitive":!0},performAction:u(function(c,i,y,b){function r(){let n=i.yytext.indexOf("%%");if(n===0)return!1;if(n>0){let F=i.yytext.slice(0,n),B=i.yytext.slice(n);B&&c.lexer.unput(B),i.yytext=F}return!0}u(r,"processId");var H=b;switch(y){case 0:return 38;case 1:return 40;case 2:return 39;case 3:return 44;case 4:return 51;case 5:return 52;case 6:return 53;case 7:return 54;case 8:return 5;case 9:break;case 10:break;case 11:break;case 12:break;case 13:return this.pushState("SCALE"),17;break;case 14:return 18;case 15:this.popState();break;case 16:return this.begin("acc_title"),33;break;case 17:return this.popState(),"acc_title_value";break;case 18:return this.begin("acc_descr"),35;break;case 19:return this.popState(),"acc_descr_value";break;case 20:this.begin("acc_descr_multiline");break;case 21:this.popState();break;case 22:return"acc_descr_multiline_value";case 23:return this.pushState("CLASSDEF"),41;break;case 24:return this.popState(),this.pushState("CLASSDEFID"),"DEFAULT_CLASSDEF_ID";break;case 25:return this.popState(),this.pushState("CLASSDEFID"),42;break;case 26:return this.popState(),43;break;case 27:return this.pushState("CLASS"),48;break;case 28:return this.popState(),this.pushState("CLASS_STYLE"),49;break;case 29:return this.popState(),50;break;case 30:return this.pushState("STYLE"),45;break;case 31:return this.popState(),this.pushState("STYLEDEF_STYLES"),46;break;case 32:return this.popState(),47;break;case 33:return this.pushState("SCALE"),17;break;case 34:return 18;case 35:this.popState();break;case 36:this.pushState("STATE");break;case 37:return this.popState(),i.yytext=i.yytext.slice(0,-8).trim(),25;break;case 38:return this.popState(),i.yytext=i.yytext.slice(0,-8).trim(),26;break;case 39:return this.popState(),i.yytext=i.yytext.slice(0,-10).trim(),27;break;case 40:return this.popState(),i.yytext=i.yytext.slice(0,-8).trim(),25;break;case 41:return this.popState(),i.yytext=i.yytext.slice(0,-8).trim(),26;break;case 42:return this.popState(),i.yytext=i.yytext.slice(0,-10).trim(),27;break;case 43:return 51;case 44:return 52;case 45:return 53;case 46:return 54;case 47:this.pushState("STATE_STRING");break;case 48:return this.pushState("STATE_ID"),"AS";break;case 49:if(!r())return;return this.popState(),"ID";break;case 50:this.popState();break;case 51:return"STATE_DESCR";case 52:return 19;case 53:this.popState();break;case 54:return this.popState(),this.pushState("struct"),20;break;case 55:return this.popState(),21;break;case 56:break;case 57:return this.begin("NOTE"),29;break;case 58:return this.popState(),this.pushState("NOTE_ID"),59;break;case 59:return this.popState(),this.pushState("NOTE_ID"),60;break;case 60:this.popState(),this.pushState("FLOATING_NOTE");break;case 61:return this.popState(),this.pushState("FLOATING_NOTE_ID"),"AS";break;case 62:break;case 63:return"NOTE_TEXT";case 64:if(!r())return;return this.popState(),"ID";break;case 65:if(!r())return;return this.popState(),this.pushState("NOTE_TEXT"),24;break;case 66:return this.popState(),i.yytext=i.yytext.substr(2).trim(),31;break;case 67:return this.popState(),i.yytext=i.yytext.slice(0,-8).trim(),31;break;case 68:return 6;case 69:return 6;case 70:return 16;case 71:return 57;case 72:return r()?24:void 0;case 73:return i.yytext=i.yytext.trim(),14;break;case 74:return 15;case 75:return 28;case 76:return 58;case 77:return 5;case 78:return"INVALID"}},"anonymous"),rules:[/^(?:click\b)/i,/^(?:href\b)/i,/^(?:"[^"]*")/i,/^(?:default\b)/i,/^(?:.*direction\s+TB[^\n]*)/i,/^(?:.*direction\s+BT[^\n]*)/i,/^(?:.*direction\s+RL[^\n]*)/i,/^(?:.*direction\s+LR[^\n]*)/i,/^(?:[\n]+)/i,/^(?:[\s]+)/i,/^(?:((?!\n)\s)+)/i,/^(?:#[^\n]*)/i,/^(?:%%(?!\{)[^\n]*)/i,/^(?:scale\s+)/i,/^(?:\d+)/i,/^(?:\s+width\b)/i,/^(?:accTitle\s*:\s*)/i,/^(?:(?!\n||)*[^\n]*)/i,/^(?:accDescr\s*:\s*)/i,/^(?:(?!\n||)*[^\n]*)/i,/^(?:accDescr\s*\{\s*)/i,/^(?:[\}])/i,/^(?:[^\}]*)/i,/^(?:classDef\s+)/i,/^(?:DEFAULT\s+)/i,/^(?:\w+\s+)/i,/^(?:[^\n]*)/i,/^(?:class\s+)/i,/^(?:(\w+)+((,\s*\w+)*))/i,/^(?:[^\n]*)/i,/^(?:style\s+)/i,/^(?:[\w,]+\s+)/i,/^(?:[^\n]*)/i,/^(?:scale\s+)/i,/^(?:\d+)/i,/^(?:\s+width\b)/i,/^(?:state\s+)/i,/^(?:.*<<fork>>)/i,/^(?:.*<<join>>)/i,/^(?:.*<<choice>>)/i,/^(?:.*\[\[fork\]\])/i,/^(?:.*\[\[join\]\])/i,/^(?:.*\[\[choice\]\])/i,/^(?:.*direction\s+TB[^\n]*)/i,/^(?:.*direction\s+BT[^\n]*)/i,/^(?:.*direction\s+RL[^\n]*)/i,/^(?:.*direction\s+LR[^\n]*)/i,/^(?:["])/i,/^(?:\s*as\s+)/i,/^(?:[^\n\{]*)/i,/^(?:["])/i,/^(?:[^"]*)/i,/^(?:[^\n\s\{]+)/i,/^(?:\n)/i,/^(?:\{)/i,/^(?:\})/i,/^(?:[\n])/i,/^(?:note\s+)/i,/^(?:left of\b)/i,/^(?:right of\b)/i,/^(?:")/i,/^(?:\s*as\s*)/i,/^(?:["])/i,/^(?:[^"]*)/i,/^(?:[^\n]*)/i,/^(?:\s*[^:\n\s\-]+)/i,/^(?:\s*:[^:\n;]+)/i,/^(?:[\s\S]*?\n\s*end note\b)/i,/^(?:stateDiagram\s+)/i,/^(?:stateDiagram-v2\s+)/i,/^(?:hide empty description\b)/i,/^(?:\[\*\])/i,/^(?:[^:\n\s\-\{]+)/i,/^(?:\s*:(?:[^:\n;]|:[^:\n;])+)/i,/^(?:-->)/i,/^(?:--)/i,/^(?::::)/i,/^(?:$)/i,/^(?:.)/i],conditions:{LINE:{rules:[10,11,12],inclusive:!1},struct:{rules:[10,11,12,23,27,30,36,43,44,45,46,55,56,57,71,72,73,74,75,76],inclusive:!1},FLOATING_NOTE_ID:{rules:[64],inclusive:!1},FLOATING_NOTE:{rules:[61,62,63],inclusive:!1},NOTE_TEXT:{rules:[66,67],inclusive:!1},NOTE_ID:{rules:[65],inclusive:!1},NOTE:{rules:[58,59,60],inclusive:!1},STYLEDEF_STYLEOPTS:{rules:[],inclusive:!1},STYLEDEF_STYLES:{rules:[32],inclusive:!1},STYLE_IDS:{rules:[],inclusive:!1},STYLE:{rules:[31],inclusive:!1},CLASS_STYLE:{rules:[29],inclusive:!1},CLASS:{rules:[28],inclusive:!1},CLASSDEFID:{rules:[26],inclusive:!1},CLASSDEF:{rules:[24,25],inclusive:!1},acc_descr_multiline:{rules:[21,22],inclusive:!1},acc_descr:{rules:[19],inclusive:!1},acc_title:{rules:[17],inclusive:!1},SCALE:{rules:[14,15,34,35],inclusive:!1},ALIAS:{rules:[],inclusive:!1},STATE_ID:{rules:[49],inclusive:!1},STATE_STRING:{rules:[50,51],inclusive:!1},FORK_STATE:{rules:[],inclusive:!1},STATE:{rules:[10,11,12,37,38,39,40,41,42,47,48,52,53,54],inclusive:!1},ID:{rules:[10,11,12],inclusive:!1},INITIAL:{rules:[0,1,2,3,4,5,6,7,8,9,11,12,13,16,18,20,23,27,30,33,36,54,57,68,69,70,71,72,73,74,76,77,78],inclusive:!0}}};return Y}();gt.lexer=ce;function ct(){this.yy={}}return u(ct,"Parser"),ct.prototype=gt,gt.Parser=ct,new ct}();vt.parser=vt;var je=vt,fe="TB",Jt="TB",Ut="dir",X="state",K="root",Ct="relation",pe="classDef",Se="style",ye="applyClass",tt="default",qt="divider",Qt="fill:none",Zt="fill: #333",te="c",ee="markdown",se="normal",mt="rect",Dt="rectWithTitle",ge="stateStart",Te="stateEnd",Wt="divider",jt="roundedWithTitle",be="note",Ee="noteGroup",et="statediagram",ke="state",_e=`${et}-${ke}`,re="transition",me="note",De="note-edge",ve=`${re} ${De}`,Ce=`${et}-${me}`,Ae="cluster",xe=`${et}-${Ae}`,Le="cluster-alt",Ie=`${et}-${Le}`,ie="parent",ae="note",Oe="state",At="----",Ne=`${At}${ae}`,Ht=`${At}${ie}`,ne=u((t,e=Jt)=>{if(!t.doc)return e;let l=e;for(let s of t.doc)s.stmt==="dir"&&(l=s.value);return l},"getDir"),Re=u(function(t,e){return e.db.getClasses()},"getClasses"),we=u(async function(t,e,l,s){E.info("REF0:"),E.info("Drawing state diagram (v2)",e);let{securityLevel:d,state:f,layout:S}=w();s.db.extract(s.db.getRootDocV2());let g=s.db.getData(),o=Vt(e,d);g.type=s.type,g.layoutAlgorithm=S,g.nodeSpacing=f?.nodeSpacing||50,g.rankSpacing=f?.rankSpacing||50,w().look==="neo"?g.markers=["barbNeo"]:g.markers=["barb"],g.diagramId=e,await Yt(g,o);let k=8;try{(typeof s.db.getLinks=="function"?s.db.getLinks():new Map).forEach((A,m)=>{let h=typeof m=="string"?m:typeof m?.id=="string"?m.id:"";if(!h){E.warn("\u26A0\uFE0F Invalid or missing stateId from key:",JSON.stringify(m));return}let x=o.node()?.querySelectorAll("g"),D;if(x?.forEach(I=>{I.textContent?.trim()===h&&(D=I)}),!D){E.warn("\u26A0\uFE0F Could not find node matching text:",h);return}let $=D.parentNode;if(!$){E.warn("\u26A0\uFE0F Node has no parent, cannot wrap:",h);return}let L=document.createElementNS("http://www.w3.org/2000/svg","a"),P=A.url.replace(/^"+|"+$/g,"");if(L.setAttributeNS("http://www.w3.org/1999/xlink","xlink:href",P),L.setAttribute("target","_blank"),A.tooltip){let I=A.tooltip.replace(/^"+|"+$/g,"");L.setAttribute("title",I)}$.replaceChild(L,D),L.appendChild(D),E.info("\u{1F517} Wrapped node in <a> tag for:",h,A.url)})}catch(N){E.error("\u274C Error injecting clickable links:",N)}Gt.insertTitle(o,"statediagramTitleText",f?.titleTopMargin??25,s.db.getDiagramTitle()),Mt(o,k,et,f?.useMaxWidth??!0)},"draw"),He={getClasses:Re,draw:we,getDir:ne},St=new Map,M=0;function yt(t="",e=0,l="",s=At){let d=l!==null&&l.length>0?`${s}${l}`:"";return`${Oe}-${t}${d}-${e}`}u(yt,"stateDomId");var $e=u((t,e,l,s,d,f,S,g)=>{E.trace("items",e),e.forEach(o=>{switch(o.stmt){case X:Z(t,o,l,s,d,f,S,g);break;case tt:Z(t,o,l,s,d,f,S,g);break;case Ct:{Z(t,o.state1,l,s,d,f,S,g),Z(t,o.state2,l,s,d,f,S,g);let T=S==="neo",k={id:"edge"+M,start:o.state1.id,end:o.state2.id,arrowhead:"normal",arrowTypeEnd:T?"arrow_barb_neo":"arrow_barb",style:Qt,labelStyle:"",label:V.sanitizeText(o.description??"",w()),arrowheadStyle:Zt,labelpos:te,labelType:ee,thickness:se,classes:re,look:S};d.push(k),M++}break}})},"setupDoc"),zt=u((t,e=Jt)=>{let l=e;if(t.doc)for(let s of t.doc)s.stmt==="dir"&&(l=s.value);return l},"getDir");function Q(t,e,l){if(!e.id||e.id==="</join></fork>"||e.id==="</choice>")return;e.cssClasses&&(Array.isArray(e.cssCompiledStyles)||(e.cssCompiledStyles=[]),e.cssClasses.split(" ").forEach(d=>{let f=l.get(d);f&&(e.cssCompiledStyles=[...e.cssCompiledStyles??[],...f.styles])}));let s=t.find(d=>d.id===e.id);s?Object.assign(s,e):t.push(e)}u(Q,"insertOrUpdateNode");function oe(t){return t?.classes?.join(" ")??""}u(oe,"getClassesFromDbInfo");function le(t){return t?.styles??[]}u(le,"getStylesFromDbInfo");var Z=u((t,e,l,s,d,f,S,g)=>{let o=e.id,T=l.get(o),k=oe(T),N=le(T),A=w();if(E.info("dataFetcher parsedItem",e,T,N),o!=="root"){let m=mt;e.start===!0?m=ge:e.start===!1&&(m=Te),e.type!==tt&&(m=e.type),St.get(o)||St.set(o,{id:o,shape:m,description:V.sanitizeText(o,A),cssClasses:`${k} ${_e}`,cssStyles:N});let h=St.get(o);e.description&&(Array.isArray(h.description)?(h.shape=Dt,h.description.push(e.description)):h.description?.length&&h.description.length>0?(h.shape=Dt,h.description===o?h.description=[e.description]:h.description=[h.description,e.description]):(h.shape=mt,h.description=e.description),h.description=V.sanitizeTextOrArray(h.description,A)),h.description?.length===1&&h.shape===Dt&&(h.type==="group"?h.shape=jt:h.shape=mt),!h.type&&e.doc&&(E.info("Setting cluster for XCX",o,zt(e)),h.type="group",h.isGroup=!0,h.dir=zt(e),h.shape=e.type===qt?Wt:jt,h.cssClasses=`${h.cssClasses} ${xe} ${f?Ie:""}`);let x={labelStyle:"",shape:h.shape,label:h.description,cssClasses:h.cssClasses,cssCompiledStyles:[],cssStyles:h.cssStyles,id:o,dir:h.dir,domId:yt(o,M),type:h.type,isGroup:h.type==="group",padding:8,rx:10,ry:10,look:S,labelType:"markdown"};if(x.shape===Wt&&(x.label=""),t&&t.id!=="root"&&(E.trace("Setting node ",o," to be child of its parent ",t.id),x.parentId=t.id),x.centerLabel=!0,e.note){let D={labelStyle:"",shape:be,label:e.note.text,labelType:"markdown",cssClasses:Ce,cssStyles:[],cssCompiledStyles:[],id:o+Ne+"-"+M,domId:yt(o,M,ae),type:h.type,isGroup:h.type==="group",padding:A.flowchart?.padding,look:S,position:e.note.position},$=o+Ht,L={labelStyle:"",shape:Ee,label:e.note.text,cssClasses:h.cssClasses,cssStyles:[],id:o+Ht,domId:yt(o,M,ie),type:"group",isGroup:!0,padding:16,look:S,position:e.note.position};M++,L.id=$,D.parentId=$,Q(s,L,g),Q(s,D,g),Q(s,x,g);let P=o,I=D.id;e.note.position==="left of"&&(P=D.id,I=o),d.push({id:P+"-"+I,start:P,end:I,arrowhead:"none",arrowTypeEnd:"",style:Qt,labelStyle:"",classes:ve,arrowheadStyle:Zt,labelpos:te,labelType:ee,thickness:se,look:S})}else Q(s,x,g)}e.doc&&(E.trace("Adding nodes children "),$e(e,e.doc,l,s,d,!f,S,g))},"dataFetcher"),Pe=u(()=>{St.clear(),M=0},"reset"),C={START_NODE:"[*]",START_TYPE:"start",END_NODE:"[*]",END_TYPE:"end",COLOR_KEYWORD:"color",FILL_KEYWORD:"fill",BG_FILL:"bgFill",STYLECLASS_SEP:","},Kt=u(()=>new Map,"newClassesList"),Xt=u(()=>({relations:[],states:new Map,documents:{}}),"newDoc"),pt=u(t=>JSON.parse(JSON.stringify(t)),"clone"),ze=class{constructor(t){this.version=t,this.nodes=[],this.edges=[],this.rootDoc=[],this.classes=Kt(),this.documents={root:Xt()},this.currentDocument=this.documents.root,this.startEndCount=0,this.dividerCnt=0,this.links=new Map,this.getAccTitle=Rt,this.setAccTitle=Nt,this.getAccDescription=$t,this.setAccDescription=wt,this.setDiagramTitle=Pt,this.getDiagramTitle=Ft,this.clear(),this.setRootDoc=this.setRootDoc.bind(this),this.getDividerId=this.getDividerId.bind(this),this.setDirection=this.setDirection.bind(this),this.trimColon=this.trimColon.bind(this)}static{u(this,"StateDB")}static{this.relationType={AGGREGATION:0,EXTENSION:1,COMPOSITION:2,DEPENDENCY:3}}extract(t){this.clear(!0);for(let s of Array.isArray(t)?t:t.doc)switch(s.stmt){case X:this.addState(s.id.trim(),s.type,s.doc,s.description,s.note);break;case Ct:this.addRelation(s.state1,s.state2,s.description);break;case pe:this.addStyleClass(s.id.trim(),s.classes);break;case Se:this.handleStyleDef(s);break;case ye:this.setCssClass(s.id.trim(),s.styleClass);break;case"click":this.addLink(s.id,s.url,s.tooltip);break}let e=this.getStates(),l=w();Pe(),Z(void 0,this.getRootDocV2(),e,this.nodes,this.edges,!0,l.look,this.classes);for(let s of this.nodes)if(Array.isArray(s.label)){if(s.description=s.label.slice(1),s.isGroup&&s.description.length>0)throw new Error(`Group nodes can only have label. Remove the additional description for node [${s.id}]`);s.label=s.label[0]}}handleStyleDef(t){let e=t.id.trim().split(","),l=t.styleClass.split(",");for(let s of e){let d=this.getState(s);if(!d){let f=s.trim();this.addState(f),d=this.getState(f)}d&&(d.styles=l.map(f=>f.replace(/;/g,"")?.trim()))}}setRootDoc(t){E.info("Setting root doc",t),this.rootDoc=t,this.version===1?this.extract(t):this.extract(this.getRootDocV2())}docTranslator(t,e,l){if(e.stmt===Ct){this.docTranslator(t,e.state1,!0),this.docTranslator(t,e.state2,!1);return}if(e.stmt===X&&(e.id===C.START_NODE?(e.id=t.id+(l?"_start":"_end"),e.start=l):e.id=e.id.trim()),e.stmt!==K&&e.stmt!==X||!e.doc)return;let s=[],d=[];for(let f of e.doc)if(f.type===qt){let S=pt(f);S.doc=pt(d),s.push(S),d=[]}else d.push(f);if(s.length>0&&d.length>0){let f={stmt:X,id:Bt(),type:"divider",doc:pt(d)};s.push(pt(f)),e.doc=s}e.doc.forEach(f=>this.docTranslator(e,f,!0))}getRootDocV2(){return this.docTranslator({id:K,stmt:K},{id:K,stmt:K,doc:this.rootDoc},!0),{id:K,doc:this.rootDoc}}addState(t,e=tt,l=void 0,s=void 0,d=void 0,f=void 0,S=void 0,g=void 0){let o=t?.trim();if(!this.currentDocument.states.has(o))E.info("Adding state ",o,s),this.currentDocument.states.set(o,{stmt:X,id:o,descriptions:[],type:e,doc:l,note:d,classes:[],styles:[],textStyles:[]});else{let T=this.currentDocument.states.get(o);if(!T)throw new Error(`State not found: ${o}`);T.doc||(T.doc=l),T.type||(T.type=e)}if(s&&(E.info("Setting state description",o,s),(Array.isArray(s)?s:[s]).forEach(k=>this.addDescription(o,k.trim()))),d){let T=this.currentDocument.states.get(o);if(!T)throw new Error(`State not found: ${o}`);T.note=d,T.note.text=V.sanitizeText(T.note.text,w())}f&&(E.info("Setting state classes",o,f),(Array.isArray(f)?f:[f]).forEach(k=>this.setCssClass(o,k.trim()))),S&&(E.info("Setting state styles",o,S),(Array.isArray(S)?S:[S]).forEach(k=>this.setStyle(o,k.trim()))),g&&(E.info("Setting state styles",o,S),(Array.isArray(g)?g:[g]).forEach(k=>this.setTextStyle(o,k.trim())))}clear(t){this.nodes=[],this.edges=[],this.documents={root:Xt()},this.currentDocument=this.documents.root,this.startEndCount=0,this.classes=Kt(),t||(this.links=new Map,Ot())}getState(t){return this.currentDocument.states.get(t)}getStates(){return this.currentDocument.states}logDocuments(){E.info("Documents = ",this.documents)}getRelations(){return this.currentDocument.relations}addLink(t,e,l){this.links.set(t,{url:e,tooltip:l}),E.warn("Adding link",t,e,l)}getLinks(){return this.links}startIdIfNeeded(t=""){return t===C.START_NODE?(this.startEndCount++,`${C.START_TYPE}${this.startEndCount}`):t}startTypeIfNeeded(t="",e=tt){return t===C.START_NODE?C.START_TYPE:e}endIdIfNeeded(t=""){return t===C.END_NODE?(this.startEndCount++,`${C.END_TYPE}${this.startEndCount}`):t}endTypeIfNeeded(t="",e=tt){return t===C.END_NODE?C.END_TYPE:e}addRelationObjs(t,e,l=""){let s=this.startIdIfNeeded(t.id.trim()),d=this.startTypeIfNeeded(t.id.trim(),t.type),f=this.startIdIfNeeded(e.id.trim()),S=this.startTypeIfNeeded(e.id.trim(),e.type);this.addState(s,d,t.doc,t.description,t.note,t.classes,t.styles,t.textStyles),this.addState(f,S,e.doc,e.description,e.note,e.classes,e.styles,e.textStyles),this.currentDocument.relations.push({id1:s,id2:f,relationTitle:V.sanitizeText(l,w())})}addRelation(t,e,l){if(typeof t=="object"&&typeof e=="object")this.addRelationObjs(t,e,l);else if(typeof t=="string"&&typeof e=="string"){let s=this.startIdIfNeeded(t.trim()),d=this.startTypeIfNeeded(t),f=this.endIdIfNeeded(e.trim()),S=this.endTypeIfNeeded(e);this.addState(s,d),this.addState(f,S),this.currentDocument.relations.push({id1:s,id2:f,relationTitle:l?V.sanitizeText(l,w()):void 0})}}addDescription(t,e){let l=this.currentDocument.states.get(t),s=e.startsWith(":")?e.replace(":","").trim():e;l?.descriptions?.push(V.sanitizeText(s,w()))}cleanupLabel(t){return t.startsWith(":")?t.slice(2).trim():t.trim()}getDividerId(){return this.dividerCnt++,`divider-id-${this.dividerCnt}`}addStyleClass(t,e=""){this.classes.has(t)||this.classes.set(t,{id:t,styles:[],textStyles:[]});let l=this.classes.get(t);e&&l&&e.split(C.STYLECLASS_SEP).forEach(s=>{let d=s.replace(/([^;]*);/,"$1").trim();if(RegExp(C.COLOR_KEYWORD).exec(s)){let S=d.replace(C.FILL_KEYWORD,C.BG_FILL).replace(C.COLOR_KEYWORD,C.FILL_KEYWORD);l.textStyles.push(S)}l.styles.push(d)})}getClasses(){return this.classes}setCssClass(t,e){t.split(",").forEach(l=>{let s=this.getState(l);if(!s){let d=l.trim();this.addState(d),s=this.getState(d)}s?.classes?.push(e)})}setStyle(t,e){this.getState(t)?.styles?.push(e)}setTextStyle(t,e){this.getState(t)?.textStyles?.push(e)}getDirectionStatement(){return this.rootDoc.find(t=>t.stmt===Ut)}getDirection(){return this.getDirectionStatement()?.value??fe}setDirection(t){let e=this.getDirectionStatement();e?e.value=t:this.rootDoc.unshift({stmt:Ut,value:t})}trimColon(t){return t.startsWith(":")?t.slice(1).trim():t.trim()}getData(){let t=w();return{nodes:this.nodes,edges:this.edges,other:{},config:t,direction:ne(this.getRootDocV2())}}getConfig(){return w().state}},Fe=u(t=>`
defs [id$="-barbEnd"] {
    fill: ${t.transitionColor};
    stroke: ${t.transitionColor};
  }
g.stateGroup text {
  fill: ${t.nodeBorder};
  stroke: none;
  font-size: 10px;
}
g.stateGroup text {
  fill: ${t.textColor};
  stroke: none;
  font-size: 10px;

}
g.stateGroup .state-title {
  font-weight: bolder;
  fill: ${t.stateLabelColor};
}

g.stateGroup rect {
  fill: ${t.mainBkg};
  stroke: ${t.nodeBorder};
}

g.stateGroup line {
  stroke: ${t.lineColor};
  stroke-width: ${t.strokeWidth||1};
}

.transition {
  stroke: ${t.transitionColor};
  stroke-width: ${t.strokeWidth||1};
  fill: none;
}

.stateGroup .composit {
  fill: ${t.background};
  border-bottom: 1px
}

.stateGroup .alt-composit {
  fill: #e0e0e0;
  border-bottom: 1px
}

.state-note {
  stroke: ${t.noteBorderColor};
  fill: ${t.noteBkgColor};

  text {
    fill: ${t.noteTextColor};
    stroke: none;
    font-size: 10px;
  }
}

.stateLabel .box {
  stroke: none;
  stroke-width: 0;
  fill: ${t.mainBkg};
  opacity: 0.5;
}

.edgeLabel .label rect {
  fill: ${t.labelBackgroundColor};
  opacity: 0.5;
}
.edgeLabel {
  background-color: ${t.edgeLabelBackground};
  p {
    background-color: ${t.edgeLabelBackground};
  }
  rect {
    opacity: 0.5;
    background-color: ${t.edgeLabelBackground};
    fill: ${t.edgeLabelBackground};
  }
  text-align: center;
}
.edgeLabel .label text {
  fill: ${t.transitionLabelColor||t.tertiaryTextColor};
}
.label div .edgeLabel {
  color: ${t.transitionLabelColor||t.tertiaryTextColor};
}

.stateLabel text {
  fill: ${t.stateLabelColor};
  font-size: 10px;
  font-weight: bold;
}

.node circle.state-start {
  fill: ${t.specialStateColor};
  stroke: ${t.specialStateColor};
}

.node .fork-join {
  fill: ${t.specialStateColor};
  stroke: ${t.specialStateColor};
}

.node circle.state-end {
  fill: ${t.innerEndBackground};
  stroke: ${t.background};
  stroke-width: 1.5
}
.end-state-inner {
  fill: ${t.compositeBackground||t.background};
  // stroke: ${t.background};
  stroke-width: 1.5
}

.node rect {
  fill: ${t.stateBkg||t.mainBkg};
  stroke: ${t.stateBorder||t.nodeBorder};
  stroke-width: ${t.strokeWidth||1}px;
}
.node polygon {
  fill: ${t.mainBkg};
  stroke: ${t.stateBorder||t.nodeBorder};;
  stroke-width: ${t.strokeWidth||1}px;
}
[id$="-barbEnd"] {
  fill: ${t.lineColor};
}

.statediagram-cluster rect {
  fill: ${t.compositeTitleBackground};
  stroke: ${t.stateBorder||t.nodeBorder};
  stroke-width: ${t.strokeWidth||1}px;
}

.cluster-label, .nodeLabel {
  color: ${t.stateLabelColor};
  // line-height: 1;
}

.statediagram-cluster rect.outer {
  rx: 5px;
  ry: 5px;
}
.statediagram-state .divider {
  stroke: ${t.stateBorder||t.nodeBorder};
}

.statediagram-state .title-state {
  rx: 5px;
  ry: 5px;
}
.statediagram-cluster.statediagram-cluster .inner {
  fill: ${t.compositeBackground||t.background};
}
.statediagram-cluster.statediagram-cluster-alt .inner {
  fill: ${t.altBackground?t.altBackground:"#efefef"};
}

.statediagram-cluster .inner {
  rx:0;
  ry:0;
}

.statediagram-state rect.basic {
  rx: 5px;
  ry: 5px;
}
.statediagram-state rect.divider {
  stroke-dasharray: 10,10;
  fill: ${t.altBackground?t.altBackground:"#efefef"};
}

.note-edge {
  stroke-dasharray: 5;
}

.statediagram-note rect {
  fill: ${t.noteBkgColor};
  stroke: ${t.noteBorderColor};
  stroke-width: 1px;
  rx: 0;
  ry: 0;
}
.statediagram-note rect {
  fill: ${t.noteBkgColor};
  stroke: ${t.noteBorderColor};
  stroke-width: 1px;
  rx: 0;
  ry: 0;
}

.statediagram-note text {
  fill: ${t.noteTextColor};
}

.statediagram-note .nodeLabel {
  color: ${t.noteTextColor};
}
.statediagram .edgeLabel {
  color: red; // ${t.noteTextColor};
}

[id$="-dependencyStart"], [id$="-dependencyEnd"] {
  fill: ${t.lineColor};
  stroke: ${t.lineColor};
  stroke-width: ${t.strokeWidth||1};
}

.statediagramTitleText {
  text-anchor: middle;
  font-size: 18px;
  fill: ${t.textColor};
}

[data-look="neo"].statediagram-cluster rect {
  fill: ${t.mainBkg};
  stroke: ${t.useGradient?"url("+t.svgId+"-gradient)":t.stateBorder||t.nodeBorder};
  stroke-width: ${t.strokeWidth??1};
}
[data-look="neo"].statediagram-cluster rect.outer {
  rx: ${t.radius}px;
  ry: ${t.radius}px;
  filter: ${t.dropShadow?t.dropShadow.replace("url(#drop-shadow)",`url(${t.svgId}-drop-shadow)`):"none"}
}
`,"getStyles"),Ke=Fe;export{je as a,He as b,ze as c,Ke as d};
