import {Controller} from "@nestjs/common"
import {AiService} from './ai.service'
@Controller("api/ai")

export class AiController {
    constructor(private ai: AiService){}


}